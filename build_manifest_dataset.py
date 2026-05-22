#!/usr/bin/env python3
"""Build an ImageFolder-compatible dataset from a base dataset and a synthetic manifest."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATASET_CACHE_NAMES = {"ham_cache_u8.pt", "cache_uint8_256.pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone a base dataset and add synthetic train images from manifest.csv.")
    parser.add_argument("--src-dataset", required=True, help="Base ImageFolder dataset with train/valid/test")
    parser.add_argument("--manifest", required=True, help="Feature-aware GAN-mix manifest.csv")
    parser.add_argument("--out-dataset", required=True, help="Output dataset directory")
    parser.add_argument("--link-mode", default="hardlink", choices=["hardlink", "copy", "symlink"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(f, delimiter=delimiter))


def safe_remove(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {path}")
        safe_remove(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def clone_tree(src_root: Path, dst_root: Path, mode: str) -> None:
    for root, _, files in os.walk(src_root):
        src_dir = Path(root)
        dst_dir = dst_root / src_dir.relative_to(src_root)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            if file_name in DATASET_CACHE_NAMES:
                continue
            src = src_dir / file_name
            dst = dst_dir / file_name
            link_or_copy_file(src, dst, mode)


def remove_dataset_caches(dataset_root: Path) -> None:
    for cache_name in DATASET_CACHE_NAMES:
        for path in dataset_root.rglob(cache_name):
            path.unlink(missing_ok=True)


def build_dataset(src_dataset: Path, manifest_path: Path, out_dataset: Path, link_mode: str, overwrite: bool) -> None:
    if not (src_dataset / "train").exists():
        raise FileNotFoundError(f"Base dataset must contain train/: {src_dataset}")

    ensure_empty_dir(out_dataset, overwrite)
    clone_tree(src_dataset, out_dataset, link_mode)
    remove_dataset_caches(out_dataset)

    rows = read_manifest(manifest_path)
    manifest_rows = []
    for idx, row in enumerate(rows):
        src = Path(row["image_path"])
        label = row["label"]
        if not src.exists():
            raise FileNotFoundError(f"Manifest image does not exist: {src}")
        if src.suffix.lower() not in IMG_EXTS:
            continue

        dst_name = f"feature_gan__{row.get('selection_mode', 'selected')}__{label}__{idx:06d}{src.suffix.lower()}"
        dst = out_dataset / "train" / label / dst_name
        link_or_copy_file(src, dst, link_mode)

        out_row = dict(row)
        out_row["dst_path"] = str(dst)
        manifest_rows.append(out_row)

    out_manifest = out_dataset / "_feature_aware_manifest.csv"
    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    if fieldnames:
        with out_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"[SUCCESS] Built dataset: {out_dataset.resolve()}")
    print(f"[*] Added synthetic images: {len(manifest_rows)}")
    print("[*] Dataset tensor caches were not copied; train.py will rebuild cache from the final image set.")


def main() -> None:
    args = parse_args()
    build_dataset(Path(args.src_dataset), Path(args.manifest), Path(args.out_dataset), args.link_mode, args.overwrite)


if __name__ == "__main__":
    main()
