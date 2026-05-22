#!/usr/bin/env python3
"""Export SupCon feature embeddings for real or synthetic images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from model import JointSkinLesionClassifier


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embeddings from a SupCon-trained classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", required=True, help="Directory for embeddings.npy and metadata.csv")
    parser.add_argument("--split-csv", default="", help="CSV with image_path,label[,split,source]")
    parser.add_argument("--image-root", default="", help="Root used for relative image_path values in --split-csv")
    parser.add_argument("--image-dir", default="", help="ImageFolder directory, e.g. dataset/train")
    parser.add_argument("--split", default="", help="Split name written to metadata if split-csv has no split column")
    parser.add_argument("--source", default="real", choices=["real", "synthetic"], help="Source written to metadata")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key.replace("_orig_mod.", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def load_model(checkpoint: Path, device: torch.device) -> JointSkinLesionClassifier:
    model = JointSkinLesionClassifier(num_classes=len(CLASS_NAMES)).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()
    return model


def build_transform():
    return v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def normalize_label(label: str | int) -> Tuple[int, str]:
    if isinstance(label, int):
        return label, CLASS_NAMES[label]

    text = str(label).strip()
    if text in CLASS_NAMES:
        return CLASS_NAMES.index(text), text

    idx = int(text)
    return idx, CLASS_NAMES[idx]


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(f, delimiter=delimiter))


class CsvImageDataset(Dataset):
    def __init__(self, rows: List[dict], image_root: Path | None, default_split: str, default_source: str):
        self.rows = rows
        self.image_root = image_root
        self.default_split = default_split
        self.default_source = default_source
        self.transform = build_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        raw_path = row.get("image_path") or row.get("path") or row.get("filepath")
        if not raw_path:
            raise KeyError("CSV must contain image_path, path, or filepath column")

        img_path = self._resolve_path(raw_path)
        label_value = row.get("label") or row.get("class") or row.get("class_name")
        if label_value is None:
            label_value = img_path.parent.name
        label_idx, label_name = normalize_label(label_value)

        with Image.open(img_path) as img:
            image = self.transform(img.convert("RGB"))

        meta = {
            "image_path": str(img_path),
            "label": label_name,
            "split": row.get("split") or self.default_split,
            "source": row.get("source") or self.default_source,
        }
        return image, label_idx, meta


class ImageFolderWithMeta(Dataset):
    def __init__(self, image_dir: Path, split: str, source: str):
        self.dataset = datasets.ImageFolder(str(image_dir), transform=build_transform())
        self.split = split
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label_idx = self.dataset[idx]
        path, _ = self.dataset.samples[idx]
        label_name = self.dataset.classes[label_idx]
        meta = {
            "image_path": str(path),
            "label": label_name,
            "split": self.split,
            "source": self.source,
        }
        return image, label_idx, meta


def collate_batch(batch):
    images, labels, metas = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(metas)


def make_dataset(args: argparse.Namespace) -> Dataset:
    if args.split_csv:
        image_root = Path(args.image_root) if args.image_root else None
        return CsvImageDataset(read_csv_rows(Path(args.split_csv)), image_root, args.split, args.source)

    if args.image_dir:
        split = args.split or Path(args.image_dir).name
        return ImageFolderWithMeta(Path(args.image_dir), split=split, source=args.source)

    raise ValueError("Provide either --split-csv or --image-dir")


@torch.no_grad()
def export_embeddings(model: JointSkinLesionClassifier, loader: DataLoader, device: torch.device):
    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_rows: List[dict] = []

    for images, labels, metas in tqdm(loader, desc="Export embeddings"):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        logits, embeddings = model(images, return_embeddings=True)
        probs = torch.softmax(logits.float(), dim=1)
        conf, pred = torch.max(probs, dim=1)

        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.append(labels.numpy())

        for meta, pred_idx, confidence in zip(metas, pred.cpu().numpy(), conf.cpu().numpy()):
            row = dict(meta)
            row["predicted_label"] = CLASS_NAMES[int(pred_idx)]
            row["confidence"] = f"{float(confidence):.8f}"
            all_rows.append(row)

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0), all_rows


def write_outputs(output_dir: Path, embeddings: np.ndarray, labels: np.ndarray, rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)

    with (output_dir / "image_paths.txt").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["image_path"] + "\n")

    fieldnames = ["image_path", "label", "split", "source", "predicted_label", "confidence"]
    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = make_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = load_model(Path(args.checkpoint), device)
    embeddings, labels, rows = export_embeddings(model, loader, device)
    write_outputs(Path(args.output_dir), embeddings, labels, rows)
    print(f"[SUCCESS] Exported {len(rows)} embeddings to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
