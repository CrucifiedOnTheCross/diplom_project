#!/usr/bin/env python3
"""Package experiment reports and source context into one downloadable archive."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "dataset",
    "dataset_augmented",
    "dataset_preprocessed",
    "dataset_preprocessed_v2",
    "datasets",
    "datasets_gan_mix",
    "datasets_gan_mix_raw",
    "datasets_feature_aware",
    "gan_data",
    "gan_data_raw",
    "gan_data_raw_zips",
    "gan_training_runs",
    "gan_training_runs_batch32",
    "gan_training_runs_quality",
    "gan_training_runs_transfer",
    "science_folder",
    "synthetic_mixing_runs",
    "stylegan3-brecahad",
    "stylegan2-ada-pytorch",
    "embedding_analysis",
    "feature_aware_report",
    "diploma_results",
    "diploma_results__full",
    "diploma_figures",
    "gradcam_results",
    "gradcam_diploma",
    "threshold_analysis_20_raw_supcon",
    "binary_threshold_analysis_20_raw_supcon",
    "report_runs",
    "audit_md",
    "audit_md_fast",
}

HEAVY_EXTS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".onnx",
    ".h5",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Create a downloadable experiment archive with code context.")
    parser.add_argument("--output", default=f"experiment_results_{stamp}.tar.gz")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--include-gradcam-images", action="store_true")
    parser.add_argument("--include-diploma-figures", action="store_true")
    parser.add_argument("--include-embedding-plots", action="store_true")
    parser.add_argument("--max-file-mb", type=float, default=25.0)
    return parser.parse_args()


def should_skip_code_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & DEFAULT_EXCLUDE_DIRS:
        return True
    if path.suffix.lower() in HEAVY_EXTS:
        return True
    return False


def copy_file(src: Path, dst: Path, copied: List[dict], category: str) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"category": category, "source": str(src), "archive_path": str(dst), "size_bytes": src.stat().st_size})


def copy_tree_filtered(
    src_root: Path,
    dst_root: Path,
    copied: List[dict],
    category: str,
    include_patterns: Sequence[str],
    exclude_exts: set[str] | None = None,
    max_file_bytes: int | None = None,
) -> None:
    if not src_root.exists():
        return
    exclude_exts = exclude_exts or set()

    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            src = root_path / name
            rel = src.relative_to(src_root)
            if src.suffix.lower() in exclude_exts:
                continue
            if max_file_bytes is not None and src.stat().st_size > max_file_bytes:
                continue
            if include_patterns and not any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(str(rel), pat) for pat in include_patterns):
                continue
            copy_file(src, dst_root / rel, copied, category)


def list_git_tracked_files() -> List[Path]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def copy_code_snapshot(bundle_dir: Path, copied: List[dict]) -> None:
    tracked = list_git_tracked_files()
    if tracked:
        for rel in tracked:
            if should_skip_code_path(rel):
                continue
            copy_file(rel, bundle_dir / "code_snapshot" / rel, copied, "code_snapshot")
        return

    for root, dirs, files in os.walk("."):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDE_DIRS and not d.startswith("audit_md")]
        for name in files:
            rel = (root_path / name).relative_to(".")
            if should_skip_code_path(rel):
                continue
            copy_file(rel, bundle_dir / "code_snapshot" / rel, copied, "code_snapshot")


def copy_root_reports(bundle_dir: Path, copied: List[dict]) -> None:
    patterns = [
        "*.csv",
        "*.png",
        "*.md",
        "*.txt",
        "*.log",
    ]
    for src in sorted(Path(".").iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() in HEAVY_EXTS:
            continue
        if any(fnmatch.fnmatch(src.name, pat) for pat in patterns):
            copy_file(src, bundle_dir / "root_reports" / src.name, copied, "root_reports")


def copy_science_reports(science_dir: Path, bundle_dir: Path, copied: List[dict], include_checkpoints: bool, max_file_bytes: int) -> None:
    patterns = [
        "*.txt",
        "*.csv",
        "*.json",
        "*.png",
    ]
    exclude_exts = set()
    if not include_checkpoints:
        exclude_exts.update({".pth", ".pt", ".ckpt"})
    copy_tree_filtered(science_dir, bundle_dir / "science_folder", copied, "science_folder", patterns, exclude_exts, max_file_bytes)


def copy_known_report_dirs(bundle_dir: Path, copied: List[dict], args: argparse.Namespace, max_file_bytes: int) -> None:
    report_dirs = [
        ("gan_analysis", ["*.csv", "*.md", "*.sh", "*.json", "*.jsonl", "*.txt", "*.png"]),
        ("feature_aware_report", ["*.csv", "*.png", "*.txt", "*.md"]),
        ("embedding_analysis", ["*.csv", "*.txt", "*.png"] if args.include_embedding_plots else ["*.csv", "*.txt"]),
        ("report_runs", ["*.csv", "*.txt", "*.log"]),
        ("datasets_gan_mix_raw", ["*_summary.csv", "*_manifest.csv", "_global_gan_dataset_summary.csv"]),
        ("datasets_feature_aware", ["*_manifest.csv", "*_summary.csv"]),
        ("synthetic_mixing_runs", ["manifest.csv", "*_summary.csv", "all_synthetic.csv"]),
    ]
    for name, patterns in report_dirs:
        copy_tree_filtered(Path(name), bundle_dir / name, copied, name, patterns, HEAVY_EXTS, max_file_bytes)

    gradcam_patterns = ["*.csv", "*.md", "*.txt"]
    if args.include_gradcam_images:
        gradcam_patterns.append("*.png")
    copy_tree_filtered(Path("gradcam_diploma"), bundle_dir / "gradcam_diploma", copied, "gradcam_diploma", gradcam_patterns, HEAVY_EXTS, max_file_bytes)
    copy_tree_filtered(Path("gradcam_results"), bundle_dir / "gradcam_results", copied, "gradcam_results", gradcam_patterns, HEAVY_EXTS, max_file_bytes)

    if args.include_diploma_figures:
        copy_tree_filtered(Path("diploma_figures"), bundle_dir / "diploma_figures", copied, "diploma_figures", ["*.png", "*.csv", "*.md", "*.txt"], HEAVY_EXTS, max_file_bytes)


def copy_stylegan_logs(bundle_dir: Path, copied: List[dict], max_file_bytes: int) -> None:
    patterns = ["metric-fid50k_full.jsonl", "stats.jsonl", "training_options.json", "log.txt", "fid*.jsonl"]
    for run_root in sorted(Path(".").glob("gan_training_runs*")):
        if run_root.is_dir():
            copy_tree_filtered(run_root, bundle_dir / "stylegan_metric_logs" / run_root.name, copied, "stylegan_metric_logs", patterns, HEAVY_EXTS, max_file_bytes)


def write_archive_readme(bundle_dir: Path, args: argparse.Namespace) -> None:
    text = f"""Experiment results archive

Created from: {Path.cwd()}
Science dir: {args.science_dir}

Contents:
- code_snapshot/: tracked project code and docs for context
- root_reports/: top-level summary CSV files and plots
- science_folder/: per-experiment reports, calibration, thresholds, training plots
- gan_analysis/: FID analysis and GAN-mix plans
- feature_aware_report/: feature-aware GAN-mix comparison report
- embedding_analysis/: embedding-space CSV/TXT reports and optional plots
- datasets_gan_mix_raw/, datasets_feature_aware/: manifests and dataset summaries only
- synthetic_mixing_runs/: selection manifests only
- gradcam_diploma/, gradcam_results/: Grad-CAM tables and optional images
- stylegan_metric_logs/: StyleGAN metric/log files only

Excluded by default:
- datasets and generated images
- model checkpoints and GAN snapshots
- large archives
- external repositories

include_checkpoints={args.include_checkpoints}
include_gradcam_images={args.include_gradcam_images}
include_diploma_figures={args.include_diploma_figures}
include_embedding_plots={args.include_embedding_plots}
max_file_mb={args.max_file_mb}
"""
    (bundle_dir / "README.txt").write_text(text, encoding="utf-8")


def write_manifest(bundle_dir: Path, copied: List[dict]) -> None:
    out = bundle_dir / "archive_manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "source", "archive_path", "size_bytes"])
        writer.writeheader()
        writer.writerows(copied)


def make_tar(output: Path, bundle_dir: Path) -> None:
    with tarfile.open(output, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)


def main() -> None:
    args = parse_args()
    max_file_bytes = int(args.max_file_mb * 1024 * 1024)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output archive already exists: {output}")

    tmp_root = Path(".archive_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    bundle_dir = tmp_root / "experiment_results_bundle"
    bundle_dir.mkdir(parents=True)

    copied: List[dict] = []
    try:
        copy_code_snapshot(bundle_dir, copied)
        copy_root_reports(bundle_dir, copied)
        copy_science_reports(Path(args.science_dir), bundle_dir, copied, args.include_checkpoints, max_file_bytes)
        copy_known_report_dirs(bundle_dir, copied, args, max_file_bytes)
        copy_stylegan_logs(bundle_dir, copied, max_file_bytes)
        write_archive_readme(bundle_dir, args)
        write_manifest(bundle_dir, copied)
        make_tar(output, bundle_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[SUCCESS] Archive created: {output.resolve()}")
    print(f"[*] Files included: {len(copied)}")


if __name__ == "__main__":
    main()
