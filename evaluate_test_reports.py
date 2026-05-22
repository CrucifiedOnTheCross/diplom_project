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
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
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
        pin_memory=True,
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


@torch.no_grad()
def collect_predictions(model, loader, device_obj: torch.device):
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc="Test inference"):
        images = images.to(device_obj, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device_obj.type == "cuda"):
            logits = model(images, return_embeddings=False)
        probs = torch.softmax(logits.float(), dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


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
    for exp_dir in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        exp_name = exp_dir.name
        out_report = exp_dir / "medical_metrics_report_test.txt"
        model_path = exp_dir / "best_model.pth"
        config_path = exp_dir / "config.txt"

        if out_report.exists() and not args.force:
            print(f"[SKIP] {exp_name}: test report already exists")
            continue
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

            dataset, loader = build_loader(test_dir, args.batch_size, args.num_workers)
            model = load_model(model_path, device_obj)
            y_true, y_pred, y_probs = collect_predictions(model, loader, device_obj)
            metrics = calculate_advanced_metrics(y_true, y_pred, y_probs)
            save_medical_report(exp_dir, y_true, y_pred, y_probs, metrics, filename="medical_metrics_report_test.txt")
            rows.append({
                "experiment": exp_name,
                "status": "ok",
                "data_dir": data_dir,
                "eval_policy": eval_policy,
                "test_dir": str(test_dir),
                "num_test": len(dataset),
                "report_path": str(out_report),
                "detail": "",
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
