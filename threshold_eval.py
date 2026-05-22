#!/usr/bin/env python3
"""
threshold_eval.py

Автоматически:
1) проходит по всем папкам экспериментов;
2) загружает best_model.pth и calibration_report.json;
3) выбирает оптимальные threshold на valid ДО и ПОСЛЕ calibration;
4) применяет выбранные threshold на test;
5) сохраняет per-experiment CSV + JSON и общий summary CSV.

Поддерживаемые режимы:
- malignant : злокачественные vs доброкачественные
- melanoma  : melanoma vs all

Примеры запуска:
    python threshold_eval.py
    python threshold_eval.py --mode malignant
    python threshold_eval.py --mode melanoma --criterion youden
    python threshold_eval.py --device cpu
    python threshold_eval.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import JointSkinLesionClassifier

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL_IDX = CLASS_NAMES.index("mel")
MALIGNANT_CLASSES = {'akiec', 'bcc', 'mel'}
MALIGNANT_IDXS = [CLASS_NAMES.index(c) for c in MALIGNANT_CLASSES]


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold analysis before/after calibration")
    parser.add_argument("--experiments_dir", type=str, default="science_folder")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="malignant", choices=["malignant", "melanoma"])
    parser.add_argument(
        "--criterion",
        type=str,
        default="youden",
        choices=["youden", "mcc", "f1", "balanced_accuracy"],
        help="Критерий выбора оптимального threshold",
    )
    parser.add_argument(
        "--checkpoint_min_age_sec",
        type=int,
        default=120,
        help="Если best_model.pth слишком свежий, эксперимент считается активным и пропускается",
    )
    parser.add_argument("--force", action="store_true", help="Пересчитать даже если threshold_report уже существует")
    return parser.parse_args()


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k.replace("_orig_mod.", "")] = v
        else:
            cleaned[k] = v
    return cleaned


def load_model(model_path: Path, device: str):
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device_obj)

    state = torch.load(model_path, map_location=device_obj)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device_obj


def parse_config(config_path: Path) -> Dict[str, str]:
    cfg = {}
    if not config_path.exists():
        return cfg
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def resolve_eval_dirs(data_dir: str) -> Tuple[Path, Path, str]:
    """
    В большинстве случаев valid/test лежат в data_dir.
    Для dataset_augmented возможен fallback на dataset_preprocessed/test,
    если своих valid/test нет.
    """
    p = Path(data_dir)
    valid_dir = p / "valid"
    test_dir = p / "test"
    if valid_dir.exists() and test_dir.exists():
        return valid_dir, test_dir, f"direct:{data_dir}"

    if p.name == "dataset_augmented":
        fallback = Path("dataset_preprocessed")
        valid_dir = fallback / "valid"
        test_dir = fallback / "test"
        if valid_dir.exists() and test_dir.exists():
            return valid_dir, test_dir, f"fallback:{data_dir}->dataset_preprocessed"

    fallback = Path("dataset")
    valid_dir = fallback / "valid"
    test_dir = fallback / "test"
    if valid_dir.exists() and test_dir.exists():
        return valid_dir, test_dir, f"fallback:{data_dir}->dataset"

    raise FileNotFoundError(f"Не найдены valid/test split для data_dir={data_dir}")


def build_loader(split_dir: Path, batch_size: int, num_workers: int):
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)

    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Порядок классов не совпадает.\n"
            f"Ожидается: {CLASS_NAMES}\n"
            f"Получено:  {dataset.classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


@torch.no_grad()
def collect_binary_probs(model, loader, device_obj, mode: str, temperature: float | None = None):
    y_true_bin = []
    probs_bin = []

    temp = None
    if temperature is not None and temperature > 0:
        temp = float(temperature)

    for images, labels in loader:
        images = images.to(device_obj, non_blocking=True)
        labels = labels.to(device_obj, non_blocking=True)

        logits = model(images, return_embeddings=False)
        if temp is not None:
            logits = logits / temp

        probs = torch.softmax(logits, dim=1)

        if mode == "melanoma":
            p_bin = probs[:, MEL_IDX]
            y_bin = (labels == MEL_IDX).long()
        elif mode == "malignant":
            p_bin = probs[:, MALIGNANT_IDXS].sum(dim=1)
            y_bin = torch.isin(labels, torch.tensor(MALIGNANT_IDXS, device=labels.device)).long()
        else:
            raise ValueError(f"Неизвестный mode: {mode}")

        y_true_bin.extend(y_bin.cpu().numpy().tolist())
        probs_bin.extend(p_bin.cpu().numpy().tolist())

    return np.array(y_true_bin), np.array(probs_bin)


def evaluate_thresholds(y_true_bin: np.ndarray, probs_bin: np.ndarray) -> List[Dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    rows = []

    for thr in thresholds:
        y_pred = (probs_bin >= thr).astype(int)

        precision = precision_score(y_true_bin, y_pred, zero_division=0)
        recall = recall_score(y_true_bin, y_pred, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred, zero_division=0)
        acc = accuracy_score(y_true_bin, y_pred)
        bacc = balanced_accuracy_score(y_true_bin, y_pred)

        if len(np.unique(y_pred)) > 1:
            mcc = matthews_corrcoef(y_true_bin, y_pred)
        else:
            mcc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = recall + specificity - 1.0

        rows.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "mcc": float(mcc),
            "youden": float(youden),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })
    return rows


def evaluate_fixed_threshold(y_true_bin: np.ndarray, probs_bin: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs_bin >= threshold).astype(int)
    precision = precision_score(y_true_bin, y_pred, zero_division=0)
    recall = recall_score(y_true_bin, y_pred, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred, zero_division=0)
    acc = accuracy_score(y_true_bin, y_pred)
    bacc = balanced_accuracy_score(y_true_bin, y_pred)
    mcc = matthews_corrcoef(y_true_bin, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youden = recall + specificity - 1.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "mcc": float(mcc),
        "youden": float(youden),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_best_row(rows: List[Dict[str, float]], criterion: str) -> Dict[str, float]:
    if criterion == "youden":
        return max(rows, key=lambda x: (x["youden"], x["mcc"], x["f1"]))
    if criterion == "mcc":
        return max(rows, key=lambda x: (x["mcc"], x["youden"], x["f1"]))
    if criterion == "f1":
        return max(rows, key=lambda x: (x["f1"], x["mcc"], x["youden"]))
    if criterion == "balanced_accuracy":
        return max(rows, key=lambda x: (x["balanced_accuracy"], x["mcc"], x["youden"]))
    raise ValueError(f"Неизвестный criterion: {criterion}")


def save_threshold_rows(rows: List[Dict[str, float]], out_csv: Path):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold", "precision", "recall", "sensitivity", "specificity",
                "f1", "accuracy", "balanced_accuracy", "mcc", "youden",
                "tp", "fp", "fn", "tn"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def is_experiment_finished(exp_dir: Path) -> Tuple[bool, str]:
    if (exp_dir / "medical_metrics_report.txt").exists():
        return True, "medical_metrics_report.txt exists"
    if (exp_dir / "training_results.png").exists():
        return True, "training_results.png exists"
    return False, "final artifacts not found"


def is_checkpoint_stable(model_path: Path, min_age_sec: int) -> Tuple[bool, str]:
    if not model_path.exists():
        return False, "best_model.pth not found"
    age = time.time() - model_path.stat().st_mtime
    if age < min_age_sec:
        return False, f"checkpoint too fresh: {age:.1f}s < {min_age_sec}s"
    return True, f"checkpoint age OK: {age:.1f}s"


def load_temperature(calibration_path: Path) -> float | None:
    if not calibration_path.exists():
        return None
    try:
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        t = data.get("temperature", None)
        if t is None:
            return None
        t = float(t)
        if not math.isfinite(t) or t <= 0:
            return None
        return t
    except Exception:
        return None


def main():
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    skip_rows = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        model_path = exp_dir / "best_model.pth"
        config_path = exp_dir / "config.txt"
        calibration_path = exp_dir / "calibration_report.json"

        per_exp_json = exp_dir / f"threshold_report_{args.mode}_{args.criterion}.json"
        per_exp_csv_before = exp_dir / f"threshold_curve_before_{args.mode}.csv"
        per_exp_csv_after = exp_dir / f"threshold_curve_after_{args.mode}.csv"

        if per_exp_json.exists() and not args.force:
            print(f"[SKIP] {exp_name} | threshold report already exists")
            continue

        finished, reason_finished = is_experiment_finished(exp_dir)
        if not finished:
            print(f"[SKIP] {exp_name} | active experiment | {reason_finished}")
            skip_rows.append({
                "experiment": exp_name,
                "reason": "active experiment",
                "detail": reason_finished,
            })
            continue

        stable, reason_stable = is_checkpoint_stable(model_path, args.checkpoint_min_age_sec)
        if not stable:
            print(f"[SKIP] {exp_name} | active checkpoint | {reason_stable}")
            skip_rows.append({
                "experiment": exp_name,
                "reason": "active checkpoint",
                "detail": reason_stable,
            })
            continue

        cfg = parse_config(config_path)
        data_dir = cfg.get("data_dir", "dataset")
        try:
            valid_dir, test_dir, eval_policy = resolve_eval_dirs(data_dir)
        except Exception as e:
            print(f"[SKIP] {exp_name} | {e}")
            skip_rows.append({
                "experiment": exp_name,
                "reason": "missing test dir",
                "detail": str(e),
            })
            continue

        temperature = load_temperature(calibration_path)

        try:
            print(f"\n[START] {exp_name}")
            print(f"[*] data_dir from config: {data_dir}")
            print(f"[*] eval policy: {eval_policy}")
            print(f"[*] valid: {valid_dir}")
            print(f"[*] test: {test_dir}")

            model, device_obj = load_model(model_path, args.device)
            valid_dataset, valid_loader = build_loader(valid_dir, args.batch_size, args.num_workers)
            test_dataset, test_loader = build_loader(test_dir, args.batch_size, args.num_workers)

            # До calibration: threshold выбирается на valid, метрики считаются на test.
            y_valid_before, p_valid_before = collect_binary_probs(model, valid_loader, device_obj, args.mode, temperature=None)
            valid_rows_before = evaluate_thresholds(y_valid_before, p_valid_before)
            valid_best_before = select_best_row(valid_rows_before, args.criterion)
            save_threshold_rows(valid_rows_before, per_exp_csv_before)

            y_test_before, p_test_before = collect_binary_probs(model, test_loader, device_obj, args.mode, temperature=None)
            best_before = evaluate_fixed_threshold(y_test_before, p_test_before, valid_best_before["threshold"])

            # После calibration
            if temperature is not None:
                y_valid_after, p_valid_after = collect_binary_probs(model, valid_loader, device_obj, args.mode, temperature=temperature)
                valid_rows_after = evaluate_thresholds(y_valid_after, p_valid_after)
                valid_best_after = select_best_row(valid_rows_after, args.criterion)
                save_threshold_rows(valid_rows_after, per_exp_csv_after)

                y_test_after, p_test_after = collect_binary_probs(model, test_loader, device_obj, args.mode, temperature=temperature)
                best_after = evaluate_fixed_threshold(y_test_after, p_test_after, valid_best_after["threshold"])
            else:
                valid_best_after = None
                best_after = None

            report = {
                "experiment": exp_name,
                "mode": args.mode,
                "criterion": args.criterion,
                "threshold_selection_split": "valid",
                "metrics_split": "test",
                "model_path": str(model_path),
                "data_dir": data_dir,
                "eval_policy": eval_policy,
                "valid_dir": str(valid_dir),
                "test_dir": str(test_dir),
                "num_valid": len(valid_dataset),
                "num_test": len(test_dataset),
                "temperature": temperature,
                "valid_best_before": valid_best_before,
                "valid_best_after": valid_best_after,
                "best_before": best_before,
                "best_after": best_after,
            }

            per_exp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            row = {
                "Experiment_Folder": exp_name,
                "mode": args.mode,
                "criterion": args.criterion,
                "temperature": temperature if temperature is not None else "N/A",
                "threshold_before": best_before["threshold"],
                "sensitivity_before": best_before["sensitivity"],
                "specificity_before": best_before["specificity"],
                "precision_before": best_before["precision"],
                "f1_before": best_before["f1"],
                "mcc_before": best_before["mcc"],
                "bacc_before": best_before["balanced_accuracy"],
                "accuracy_before": best_before["accuracy"],
                "threshold_after": best_after["threshold"] if best_after else "N/A",
                "sensitivity_after": best_after["sensitivity"] if best_after else "N/A",
                "specificity_after": best_after["specificity"] if best_after else "N/A",
                "precision_after": best_after["precision"] if best_after else "N/A",
                "f1_after": best_after["f1"] if best_after else "N/A",
                "mcc_after": best_after["mcc"] if best_after else "N/A",
                "bacc_after": best_after["balanced_accuracy"] if best_after else "N/A",
                "accuracy_after": best_after["accuracy"] if best_after else "N/A",
                "delta_threshold": (best_after["threshold"] - best_before["threshold"]) if best_after else "N/A",
                "delta_sensitivity": (best_after["sensitivity"] - best_before["sensitivity"]) if best_after else "N/A",
                "delta_specificity": (best_after["specificity"] - best_before["specificity"]) if best_after else "N/A",
                "delta_precision": (best_after["precision"] - best_before["precision"]) if best_after else "N/A",
                "delta_f1": (best_after["f1"] - best_before["f1"]) if best_after else "N/A",
                "delta_mcc": (best_after["mcc"] - best_before["mcc"]) if best_after else "N/A",
                "delta_bacc": (best_after["balanced_accuracy"] - best_before["balanced_accuracy"]) if best_after else "N/A",
                "delta_accuracy": (best_after["accuracy"] - best_before["accuracy"]) if best_after else "N/A",
            }
            summary_rows.append(row)

            if best_after:
                print(
                    f"[DONE] {exp_name} | "
                    f"thr {best_before['threshold']:.3f}->{best_after['threshold']:.3f} | "
                    f"Sens {best_before['sensitivity']:.4f}->{best_after['sensitivity']:.4f} | "
                    f"Spec {best_before['specificity']:.4f}->{best_after['specificity']:.4f}"
                )
            else:
                print(f"[DONE] {exp_name} | calibration_report.json not found, only BEFORE thresholds computed")

        except Exception as e:
            print(f"[SKIP] {exp_name} | error: {e}")
            skip_rows.append({
                "experiment": exp_name,
                "reason": "runtime error",
                "detail": str(e),
            })

    # Save summary
    summary_path = experiments_dir / f"threshold_summary_{args.mode}_{args.criterion}.csv"
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[SUCCESS] Summary saved: {summary_path}")

    # Save skip log
    skip_path = experiments_dir / f"threshold_skip_log_{args.mode}_{args.criterion}.csv"
    if skip_rows:
        pd.DataFrame(skip_rows).to_csv(skip_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Skip log saved: {skip_path}")


if __name__ == "__main__":
    main()
