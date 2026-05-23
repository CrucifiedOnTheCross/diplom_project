#!/usr/bin/env python3
"""Collect reusable valid/test prediction caches for saved experiments.

This is the only reporting step that performs model inference. Downstream
medical, calibration, threshold, and significance reports should reuse the
saved CSV/NPZ files instead of loading checkpoints again.
"""

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

from model import JointSkinLesionClassifier


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FINAL_MARKERS = ["medical_metrics_report.txt", "training_results.png"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect valid/test logits and prediction CSV files.")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional experiment names to process")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-missing", action="store_true", help="Process only missing or outdated caches")
    parser.add_argument("--checkpoint-min-age-sec", type=int, default=120)
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


def infer_eval_dirs(data_dir: str) -> Tuple[Path, Path, str]:
    root = Path(data_dir)
    valid_dir = root / "valid"
    test_dir = root / "test"
    if valid_dir.exists() and test_dir.exists():
        return valid_dir, test_dir, f"direct:{root}"

    if root.name == "dataset_augmented":
        fallback = Path("dataset_preprocessed")
        valid_dir = fallback / "valid"
        test_dir = fallback / "test"
        if valid_dir.exists() and test_dir.exists():
            return valid_dir, test_dir, "fallback:dataset_augmented->dataset_preprocessed"

    fallback = Path("dataset")
    valid_dir = fallback / "valid"
    test_dir = fallback / "test"
    if valid_dir.exists() and test_dir.exists():
        return valid_dir, test_dir, f"fallback:{root}->dataset"

    raise FileNotFoundError(f"valid/test split not found for data_dir={data_dir}")


def has_final_markers(exp_dir: Path) -> bool:
    return any((exp_dir / marker).exists() for marker in FINAL_MARKERS)


def is_checkpoint_stable(model_path: Path, min_age_sec: int) -> Tuple[bool, str]:
    if not model_path.exists():
        return False, "best_model.pth not found"
    age = time.time() - model_path.stat().st_mtime
    if age < min_age_sec:
        return False, f"checkpoint too fresh: {age:.1f}s < {min_age_sec}s"
    return True, f"checkpoint age OK: {age:.1f}s"


def outputs_are_current(outputs: List[Path], dependency: Path) -> bool:
    if not dependency.exists() or not outputs or not all(path.exists() for path in outputs):
        return False
    dependency_mtime = dependency.stat().st_mtime
    return all(path.stat().st_mtime >= dependency_mtime for path in outputs)


def build_dataset(split_dir: Path) -> datasets.ImageFolder:
    transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected class order in {split_dir}: {dataset.classes}")
    return dataset


def build_loader(dataset: datasets.ImageFolder, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_model(model_path: Path, device: torch.device) -> JointSkinLesionClassifier:
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model


def image_key(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def run_inference(model, dataset, device: torch.device, batch_size: int, num_workers: int, split_name: str) -> dict:
    current_batch = batch_size
    while current_batch >= 1:
        try:
            loader = build_loader(dataset, current_batch, num_workers)
            logits_all: List[np.ndarray] = []
            labels_all: List[np.ndarray] = []
            with torch.inference_mode():
                for images, labels in tqdm(loader, desc=f"{split_name} inference bs={current_batch}"):
                    images = images.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        logits = model(images, return_embeddings=False)
                    logits_all.append(logits.float().cpu().numpy())
                    labels_all.append(labels.numpy())
            logits = np.concatenate(logits_all, axis=0)
            y_true = np.concatenate(labels_all, axis=0)
            probs = softmax_np(logits)
            y_pred = np.argmax(probs, axis=1)
            return {
                "image_path": np.array([sample[0] for sample in dataset.samples], dtype=object),
                "image_key": np.array([image_key(sample[0]) for sample in dataset.samples], dtype=object),
                "y_true": y_true.astype(np.int64),
                "logits": logits.astype(np.float32),
                "probs": probs.astype(np.float32),
                "predicted_label": y_pred.astype(np.int64),
                "class_names": np.array(CLASS_NAMES, dtype=object),
            }
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower() and current_batch > 32:
                torch.cuda.empty_cache()
                next_batch = max(32, current_batch // 2)
                print(f"[WARN] CUDA OOM on {split_name} with batch_size={current_batch}; retrying with {next_batch}")
                current_batch = next_batch
                continue
            raise

    raise RuntimeError(f"Could not run inference for {split_name}")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def write_npz(path: Path, data: dict) -> None:
    np.savez_compressed(
        path,
        image_key=data["image_key"],
        image_path=data["image_path"],
        y_true=data["y_true"],
        logits=data["logits"],
        probs=data["probs"],
        predicted_label=data["predicted_label"],
        class_names=data["class_names"],
    )


def write_predictions_csv(path: Path, data: dict) -> None:
    fieldnames = [
        "image_key",
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        *[f"prob_{cls}" for cls in CLASS_NAMES],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, image_path_value in enumerate(data["image_path"]):
            y_true = int(data["y_true"][idx])
            y_pred = int(data["predicted_label"][idx])
            probs = data["probs"][idx]
            row = {
                "image_key": str(data["image_key"][idx]),
                "image_path": str(image_path_value),
                "true_label": CLASS_NAMES[y_true],
                "predicted_label": CLASS_NAMES[y_pred],
                "confidence": float(np.max(probs)),
            }
            for cls_idx, cls in enumerate(CLASS_NAMES):
                row[f"prob_{cls}"] = float(probs[cls_idx])
            writer.writerow(row)


def process_experiment(exp_dir: Path, args: argparse.Namespace, device: torch.device) -> dict:
    exp_name = exp_dir.name
    model_path = exp_dir / "best_model.pth"
    outputs = [
        exp_dir / "valid_predictions.csv",
        exp_dir / "valid_logits.npz",
        exp_dir / "test_predictions.csv",
        exp_dir / "test_logits.npz",
    ]

    if not args.force and outputs_are_current(outputs, model_path):
        print(f"[SKIP] {exp_name}: prediction cache is up to date")
        return {"experiment": exp_name, "status": "skip", "detail": "prediction cache is up to date"}

    if not has_final_markers(exp_dir):
        print(f"[SKIP] {exp_name}: final artifacts missing")
        return {"experiment": exp_name, "status": "skip", "detail": "final artifacts missing"}

    stable, reason = is_checkpoint_stable(model_path, args.checkpoint_min_age_sec)
    if not stable:
        print(f"[SKIP] {exp_name}: {reason}")
        return {"experiment": exp_name, "status": "skip", "detail": reason}

    cfg = parse_config(exp_dir / "config.txt")
    data_dir = cfg.get("data_dir", "dataset")
    valid_dir, test_dir, eval_policy = infer_eval_dirs(data_dir)

    print(f"\n[START] {exp_name}")
    print(f"[*] data_dir: {data_dir}")
    print(f"[*] eval policy: {eval_policy}")
    print(f"[*] valid: {valid_dir}")
    print(f"[*] test: {test_dir}")

    valid_dataset = build_dataset(valid_dir)
    test_dataset = build_dataset(test_dir)
    model = load_model(model_path, device)

    valid_data = run_inference(model, valid_dataset, device, args.batch_size, args.num_workers, "valid")
    test_data = run_inference(model, test_dataset, device, args.batch_size, args.num_workers, "test")

    write_npz(exp_dir / "valid_logits.npz", valid_data)
    write_predictions_csv(exp_dir / "valid_predictions.csv", valid_data)
    write_npz(exp_dir / "test_logits.npz", test_data)
    write_predictions_csv(exp_dir / "test_predictions.csv", test_data)

    print(f"[DONE] {exp_name}: cached valid/test logits and predictions")
    return {"experiment": exp_name, "status": "ok", "detail": f"data_dir={data_dir}; policy={eval_policy}"}


def main() -> None:
    args = parse_args()
    science_dir = Path(args.science_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    selected = set(args.experiments or [])
    exp_dirs = sorted(path for path in science_dir.iterdir() if path.is_dir())
    if selected:
        exp_dirs = [path for path in exp_dirs if path.name in selected]

    rows = []
    for exp_dir in exp_dirs:
        try:
            rows.append(process_experiment(exp_dir, args, device))
        except Exception as exc:
            print(f"[ERROR] {exp_dir.name}: {exc}")
            rows.append({"experiment": exp_dir.name, "status": "error", "detail": str(exc)})
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_path = science_dir / "prediction_cache_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SUCCESS] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
