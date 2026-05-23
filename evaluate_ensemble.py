#!/usr/bin/env python3
"""Evaluate a calibrated probability ensemble of saved experiment checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from metrics import calculate_advanced_metrics, save_medical_report
from model import JointSkinLesionClassifier


CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated soft-voting ensemble on a test split.")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--ensemble-name", required=True)
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key.replace("_orig_mod.", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


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


def load_temperature(exp_dir: Path) -> float:
    path = exp_dir / "calibration_report.json"
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        temp = float(data.get("temperature", 1.0))
        return temp if np.isfinite(temp) and temp > 0 else 1.0
    except Exception:
        return 1.0


def load_model(exp_dir: Path, device_obj: torch.device):
    model_path = exp_dir / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device_obj)
    try:
        state = torch.load(model_path, map_location=device_obj, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_probs(model, loader, device_obj: torch.device, temperature: float) -> np.ndarray:
    probs_all: List[np.ndarray] = []
    for images, _ in tqdm(loader, desc="Ensemble member inference"):
        images = images.to(device_obj, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=device_obj.type == "cuda"):
            logits = model(images, return_embeddings=False)
        logits = logits.float() / max(float(temperature), 1e-6)
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs_all, axis=0)


def collect_labels(loader) -> np.ndarray:
    labels: List[int] = []
    for _, y in loader:
        labels.extend(y.numpy().tolist())
    return np.array(labels)


def write_predictions(out_dir: Path, dataset: datasets.ImageFolder, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> None:
    out_path = out_dir / "ensemble_predictions.csv"
    fieldnames = ["image_path", "true_label", "predicted_label", "confidence"] + [f"prob_{cls}" for cls in CLASS_NAMES]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (path, _) in enumerate(dataset.samples):
            row = {
                "image_path": path,
                "true_label": CLASS_NAMES[int(y_true[idx])],
                "predicted_label": CLASS_NAMES[int(y_pred[idx])],
                "confidence": float(np.max(probs[idx])),
            }
            for cls_idx, cls in enumerate(CLASS_NAMES):
                row[f"prob_{cls}"] = float(probs[idx, cls_idx])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    science_dir = Path(args.science_dir)
    out_dir = science_dir / args.ensemble_name
    out_report = out_dir / "medical_metrics_report_test.txt"

    if out_report.exists() and not args.force:
        print(f"[SKIP] Ensemble report already exists: {out_report}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset, loader = build_loader(Path(args.test_dir), args.batch_size, args.num_workers)
    y_true = collect_labels(loader)

    member_rows = []
    probs_sum = None
    for exp_name in args.experiments:
        exp_dir = science_dir / exp_name
        temp = load_temperature(exp_dir)
        print(f"\n[MEMBER] {exp_name} | temperature={temp:.6f}")
        model = load_model(exp_dir, device_obj)
        probs = collect_probs(model, loader, device_obj, temp)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        member_rows.append({"experiment": exp_name, "temperature": temp})

    avg_probs = probs_sum / len(args.experiments)
    y_pred = np.argmax(avg_probs, axis=1)
    metrics = calculate_advanced_metrics(y_true, y_pred, avg_probs)
    save_medical_report(out_dir, y_true, y_pred, avg_probs, metrics, filename="medical_metrics_report_test.txt")
    write_predictions(out_dir, dataset, y_true, y_pred, avg_probs)

    config = {
        "exp_name": args.ensemble_name,
        "type": "calibrated_probability_ensemble",
        "members": member_rows,
        "test_dir": args.test_dir,
        "num_test": len(dataset),
    }
    (out_dir / "ensemble_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.txt").write_text(
        "\n".join([
            f"exp_name: {args.ensemble_name}",
            "type: calibrated_probability_ensemble",
            f"data_dir: {Path(args.test_dir).parent}",
            f"members: {','.join(args.experiments)}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"\n[SUCCESS] Ensemble report saved to: {out_report}")


if __name__ == "__main__":
    main()
