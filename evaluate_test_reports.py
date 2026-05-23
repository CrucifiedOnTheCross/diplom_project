#!/usr/bin/env python3
"""Recompute final medical metrics on test splits from saved best_model.pth files."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from metrics import calculate_advanced_metrics, save_medical_report
from model import JointSkinLesionClassifier


CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
FINAL_MARKERS = ['medical_metrics_report.txt', 'training_results.png']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved experiment checkpoints on test splits.")
    parser.add_argument("--experiments_dir", type=str, default="science_folder")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional experiment names to evaluate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Recompute all selected experiments")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Compute only missing or outdated test reports/predictions. This is the default when --force is not used.",
    )
    parser.add_argument("--checkpoint_min_age_sec", type=int, default=120)
    return parser.parse_args()


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key.replace("_orig_mod.", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def parse_config(config_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not config_path.exists():
        return cfg
    for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
    return cfg


def infer_test_dir(data_dir: str) -> Tuple[Path, str]:
    root = Path(data_dir)
    test_dir = root / "test"
    if test_dir.exists():
        return test_dir, f"direct:{root}"

    if root.name == "dataset_augmented":
        fallback = Path("dataset_preprocessed") / "test"
        if fallback.exists():
            return fallback, "fallback:dataset_augmented->dataset_preprocessed"

    fallback = Path("dataset") / "test"
    if fallback.exists():
        return fallback, f"fallback:{root}->dataset"

    raise FileNotFoundError(f"test split not found for data_dir={data_dir}")


def has_final_markers(exp_dir: Path) -> bool:
    return any((exp_dir / marker).exists() for marker in FINAL_MARKERS)


def is_checkpoint_stable(model_path: Path, min_age_sec: int) -> Tuple[bool, str]:
    if not model_path.exists():
        return False, "best_model.pth not found"
    age = time.time() - model_path.stat().st_mtime
    if age < min_age_sec:
        return False, f"checkpoint too fresh: {age:.1f}s < {min_age_sec}s"
    return True, f"checkpoint age OK: {age:.1f}s"


def outputs_are_current(outputs: List[Path], dependencies: List[Path]) -> bool:
    if not dependencies or not all(path.exists() for path in dependencies) or not outputs:
        return False
    if not all(path.exists() for path in outputs):
        return False
    newest_dependency = max(path.stat().st_mtime for path in dependencies)
    return all(path.stat().st_mtime >= newest_dependency for path in outputs)


def build_loader(test_dir: Path, batch_size: int, num_workers: int):
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {dataset.classes}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def load_model(model_path: Path, device_obj: torch.device):
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device_obj)
    try:
        state = torch.load(model_path, map_location=device_obj, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model


def collect_predictions(model, loader, device_obj: torch.device):
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Test inference"):
            images = images.to(device_obj, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device_obj.type == "cuda"):
                logits = model(images, return_embeddings=False)
            probs = torch.softmax(logits.float(), dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def load_logits_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_predictions_csv(exp_dir: Path, cache: dict) -> Path:
    out_path = exp_dir / "test_predictions.csv"
    fieldnames = [
        "image_key",
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        *[f"prob_{cls}" for cls in CLASS_NAMES],
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        y_true = cache["y_true"].astype(int)
        y_pred = cache["predicted_label"].astype(int)
        y_probs = cache["probs"].astype(float)
        image_paths = cache["image_path"]
        image_keys = cache["image_key"] if "image_key" in cache else np.array([str(Path(p).parent.name + "/" + Path(p).name) for p in image_paths])
        for idx, image_path in enumerate(image_paths):
            row = {
                "image_key": str(image_keys[idx]),
                "image_path": str(image_path),
                "true_label": CLASS_NAMES[int(y_true[idx])],
                "predicted_label": CLASS_NAMES[int(y_pred[idx])],
                "confidence": float(np.max(y_probs[idx])),
            }
            for cls_idx, cls in enumerate(CLASS_NAMES):
                row[f"prob_{cls}"] = float(y_probs[idx, cls_idx])
            writer.writerow(row)
    return out_path


def write_summary(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "experiment", "status", "data_dir", "eval_policy", "test_dir", "num_test",
        "report_path", "detail",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiments_dir = Path(args.experiments_dir)
    device_obj = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    rows: List[dict] = []
    selected = set(args.experiments or [])
    exp_dirs = sorted(p for p in experiments_dir.iterdir() if p.is_dir())
    if selected:
        exp_dirs = [p for p in exp_dirs if p.name in selected]

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        out_report = exp_dir / "medical_metrics_report_test.txt"
        out_predictions = exp_dir / "test_predictions.csv"
        logits_path = exp_dir / "test_logits.npz"
        model_path = exp_dir / "best_model.pth"
        config_path = exp_dir / "config.txt"

        if not args.force and outputs_are_current([out_report, out_predictions], [model_path, logits_path]):
            print(f"[SKIP] {exp_name}: test predictions are up to date")
            rows.append({
                "experiment": exp_name,
                "status": "skip",
                "report_path": str(out_report),
                "detail": "test report and predictions are up to date",
            })
            continue

        if not logits_path.exists():
            print(f"[SKIP] {exp_name}: test_logits.npz missing; run collect_predictions.py first")
            rows.append({"experiment": exp_name, "status": "skip", "detail": "test_logits.npz missing"})
            continue

        if not args.force:
            print(f"[RUN] {exp_name}: missing or outdated test predictions")

        if not has_final_markers(exp_dir):
            print(f"[SKIP] {exp_name}: final artifacts missing")
            rows.append({"experiment": exp_name, "status": "skip", "detail": "final artifacts missing"})
            continue

        stable, reason = is_checkpoint_stable(model_path, args.checkpoint_min_age_sec)
        if not stable:
            print(f"[SKIP] {exp_name}: {reason}")
            rows.append({"experiment": exp_name, "status": "skip", "detail": reason})
            continue

        try:
            cfg = parse_config(config_path)
            data_dir = cfg.get("data_dir", "dataset")
            test_dir, eval_policy = infer_test_dir(data_dir)
            print(f"\n[START] {exp_name}")
            print(f"[*] data_dir: {data_dir}")
            print(f"[*] test: {test_dir} ({eval_policy})")

            cache = load_logits_cache(logits_path)
            y_true = cache["y_true"].astype(int)
            y_pred = cache["predicted_label"].astype(int)
            y_probs = cache["probs"].astype(float)
            metrics = calculate_advanced_metrics(y_true, y_pred, y_probs)
            save_medical_report(exp_dir, y_true, y_pred, y_probs, metrics, filename="medical_metrics_report_test.txt")
            predictions_path = write_predictions_csv(exp_dir, cache)
            rows.append({
                "experiment": exp_name,
                "status": "ok",
                "data_dir": data_dir,
                "eval_policy": eval_policy,
                "test_dir": str(test_dir),
                "num_test": len(y_true),
                "report_path": str(out_report),
                "detail": f"predictions={predictions_path}",
            })
            print(f"[DONE] {exp_name}: {out_report}")
        except Exception as exc:
            print(f"[ERROR] {exp_name}: {exc}")
            rows.append({"experiment": exp_name, "status": "error", "detail": str(exc)})

    summary_path = experiments_dir / "test_report_summary.csv"
    write_summary(rows, summary_path)
    print(f"\n[SUCCESS] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
