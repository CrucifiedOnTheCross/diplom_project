#!/usr/bin/env python3
"""Build final feature-space comparison across selected experiments.

The script uses only the real test split for the final geometry analysis.
It can export embeddings via export_embeddings.py and then aggregate compact
geometry metrics and publication-ready plots into one report directory.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEFAULT_EXPERIMENTS = [
    "01_base_raw",
    "ablation_raw_focal_g1_weighted",
    "24_raw_supcon_weighted",
    "gan_bcc50_akiec50_vasc25_raw_ganmix_focal_weighted",
    "33_gan_supcon_weighted",
    "confident_core_ganmix_15_supcon_weighted",
    "diverse_core_ganmix_25_supcon_weighted",
]
FOCUS_PAIRS = [("mel", "nv"), ("akiec", "bkl"), ("bcc", "bkl")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare final feature-space geometry for selected experiments.")
    parser.add_argument("--science-dir", default="science_folder")
    parser.add_argument("--test-dir", default="dataset/test", help="Real-only ImageFolder test split")
    parser.add_argument("--output-dir", default="embedding_analysis/final_feature_space_comparison")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Use existing embeddings only")
    return parser.parse_args()


def outputs_current(outputs: List[Path], dependency: Path) -> bool:
    if not dependency.exists() or not outputs or not all(path.exists() for path in outputs):
        return False
    dep_mtime = dependency.stat().st_mtime
    return all(path.stat().st_mtime >= dep_mtime for path in outputs)


def export_embeddings(exp_name: str, checkpoint: Path, test_dir: Path, exp_out: Path, args: argparse.Namespace) -> Tuple[bool, str]:
    outputs = [exp_out / "embeddings.npy", exp_out / "labels.npy", exp_out / "metadata.csv"]
    if not args.force and outputs_current(outputs, checkpoint):
        return True, "up to date"
    if args.skip_export:
        return all(path.exists() for path in outputs), "skip-export"

    exp_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "export_embeddings.py",
        "--checkpoint",
        str(checkpoint),
        "--image-dir",
        str(test_dir),
        "--split",
        "test",
        "--source",
        "real",
        "--output-dir",
        str(exp_out),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, "exported"


def label_to_name(value) -> str:
    try:
        return CLASS_NAMES[int(value)]
    except Exception:
        text = str(value)
        return text if text in CLASS_NAMES else text


def compute_projection(embeddings: np.ndarray, max_points: int, random_state: int):
    n = len(embeddings)
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    if n > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    subset = embeddings[indices]

    method = "pca"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(subset)
        method = "umap"
    except Exception:
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(5, len(subset) // 10))
            coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=random_state).fit_transform(subset)
            method = "tsne"
        except Exception:
            coords = PCA(n_components=2, random_state=random_state).fit_transform(subset)
            method = "pca"
    return coords, indices, method


def plot_by_label(exp_name: str, coords: np.ndarray, meta: pd.DataFrame, indices: np.ndarray, method: str, out_dir: Path) -> None:
    plot_meta = meta.iloc[indices].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    for cls in CLASS_NAMES:
        mask = plot_meta["label"].to_numpy() == cls
        if np.any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.78, label=cls)
    ax.set_title(f"{exp_name}: {method.upper()} by true label", fontsize=17)
    ax.set_xlabel(f"{method.upper()} 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Class", fontsize=11, title_fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_name}_{method}_by_true_label.png", dpi=300)
    plt.close(fig)


def plot_errors(exp_name: str, coords: np.ndarray, meta: pd.DataFrame, indices: np.ndarray, method: str, out_dir: Path) -> None:
    if "predicted_label" not in meta.columns:
        return
    plot_meta = meta.iloc[indices].reset_index(drop=True)
    true = plot_meta["label"].astype(str).to_numpy()
    pred = plot_meta["predicted_label"].fillna("").astype(str).to_numpy()
    has_pred = pred != ""
    errors = has_pred & (pred != true)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(coords[~errors, 0], coords[~errors, 1], s=18, alpha=0.24, label="correct")
    if np.any(errors):
        ax.scatter(coords[errors, 0], coords[errors, 1], s=38, alpha=0.9, label="error")
    ax.set_title(f"{exp_name}: {method.upper()} errors highlighted", fontsize=17)
    ax.set_xlabel(f"{method.upper()} 1", fontsize=14)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_name}_{method}_errors_highlighted.png", dpi=300)
    plt.close(fig)


def analyze_experiment(exp_name: str, exp_out: Path, figures_dir: Path, args: argparse.Namespace) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    embeddings = np.load(exp_out / "embeddings.npy")
    labels = np.load(exp_out / "labels.npy")
    meta = pd.read_csv(exp_out / "metadata.csv")
    if "label" not in meta.columns:
        meta["label"] = [label_to_name(x) for x in labels]
    else:
        meta["label"] = meta["label"].map(label_to_name)

    label_names = np.array([label_to_name(x) for x in labels])
    classes = [cls for cls in CLASS_NAMES if np.any(label_names == cls)]
    centroids = np.vstack([embeddings[label_names == cls].mean(axis=0) for cls in classes])
    centroid_dists = pairwise_distances(centroids, metric="euclidean")
    centroid_df = pd.DataFrame(centroid_dists, index=classes, columns=classes)
    centroid_df.to_csv(exp_out / "centroid_distances.csv")

    intra_rows = []
    intra_means = {}
    for cls, centroid in zip(classes, centroids):
        distances = np.linalg.norm(embeddings[label_names == cls] - centroid, axis=1)
        intra_means[cls] = float(np.mean(distances))
        intra_rows.append({
            "experiment": exp_name,
            "class": cls,
            "n": int(len(distances)),
            "intra_mean": float(np.mean(distances)),
            "intra_std": float(np.std(distances)),
            "intra_median": float(np.median(distances)),
            "intra_q75": float(np.quantile(distances, 0.75)),
        })
    intra_df = pd.DataFrame(intra_rows)
    intra_df.to_csv(exp_out / "class_distance_stats.csv", index=False)

    try:
        sil = float(silhouette_score(embeddings, labels, metric="euclidean"))
        sil_status = "ok"
    except Exception as exc:
        sil = np.nan
        sil_status = f"error: {exc}"

    nearest_rows = []
    for i, cls in enumerate(classes):
        candidates = [(classes[j], float(centroid_dists[i, j])) for j in range(len(classes)) if j != i]
        nearest, distance = min(candidates, key=lambda item: item[1])
        nearest_rows.append({"experiment": exp_name, "class": cls, "nearest_conflicting_class": nearest, "nearest_centroid_distance": distance})
    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(exp_out / "nearest_conflicting_classes.csv", index=False)

    focus_values: Dict[str, float] = {}
    for a, b in FOCUS_PAIRS:
        key = f"centroid_{a}_{b}"
        if a in classes and b in classes:
            focus_values[key] = float(centroid_dists[classes.index(a), classes.index(b)])
        else:
            focus_values[key] = np.nan

    offdiag = centroid_dists[~np.eye(len(classes), dtype=bool)]
    mean_inter = float(np.mean(offdiag))
    mean_intra = float(np.mean(list(intra_means.values())))
    inter_intra_ratio = mean_inter / mean_intra if mean_intra > 0 else np.nan

    coords, indices, method = compute_projection(embeddings, args.max_plot_points, args.random_state)
    plot_by_label(exp_name, coords, meta, indices, method, figures_dir)
    plot_errors(exp_name, coords, meta, indices, method, figures_dir)

    summary = {
        "experiment": exp_name,
        "n": int(len(labels)),
        "silhouette_score": sil,
        "silhouette_status": sil_status,
        "mean_intra_class_distance": mean_intra,
        "mean_inter_centroid_distance": mean_inter,
        "inter_intra_ratio": inter_intra_ratio,
        "projection_method": method,
    }
    summary.update(focus_values)
    for cls in ["df", "vasc"]:
        row = nearest_df[nearest_df["class"] == cls]
        summary[f"{cls}_nearest_class"] = row.iloc[0]["nearest_conflicting_class"] if not row.empty else ""
        summary[f"{cls}_nearest_distance"] = float(row.iloc[0]["nearest_centroid_distance"]) if not row.empty else np.nan

    return summary, intra_df, nearest_df


def write_markdown(summary: pd.DataFrame, skipped: pd.DataFrame, out_path: Path) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        headers = list(df.columns)
        rows = df.fillna("").astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = [
        "# Final Feature-Space Comparison",
        "",
        "Analysis uses only the real test split. Synthetic images are not included in valid/test geometry.",
        "",
        "Higher silhouette score and higher inter/intra ratio generally indicate cleaner class separation, but downstream metrics remain the primary criterion.",
        "",
        "## Summary",
        "",
        markdown_table(summary) if not summary.empty else "No experiments analyzed.",
        "",
    ]
    if not skipped.empty:
        lines.extend(["## Skipped", "", markdown_table(skipped), ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    science_dir = Path(args.science_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    embeddings_root = output_dir / "embeddings"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    intra_frames = []
    nearest_frames = []
    skipped_rows = []

    for exp_name in args.experiments:
        checkpoint = science_dir / exp_name / "best_model.pth"
        exp_out = embeddings_root / exp_name
        if not checkpoint.exists():
            skipped_rows.append({"experiment": exp_name, "reason": "missing checkpoint", "detail": str(checkpoint)})
            print(f"[SKIP] {exp_name}: missing checkpoint")
            continue

        ok, detail = export_embeddings(exp_name, checkpoint, test_dir, exp_out, args)
        if not ok:
            skipped_rows.append({"experiment": exp_name, "reason": "embedding export failed", "detail": detail})
            print(f"[SKIP] {exp_name}: {detail}")
            continue

        try:
            summary, intra_df, nearest_df = analyze_experiment(exp_name, exp_out, figures_dir, args)
            summary_rows.append(summary)
            intra_frames.append(intra_df)
            nearest_frames.append(nearest_df)
            print(f"[DONE] {exp_name}: feature-space analysis")
        except Exception as exc:
            skipped_rows.append({"experiment": exp_name, "reason": "analysis failed", "detail": str(exc)})
            print(f"[ERROR] {exp_name}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    skipped_df = pd.DataFrame(skipped_rows, columns=["experiment", "reason", "detail"])
    summary_df.to_csv(output_dir / "feature_space_summary.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(output_dir / "skipped_experiments.csv", index=False, encoding="utf-8-sig")
    if intra_frames:
        pd.concat(intra_frames, ignore_index=True).to_csv(output_dir / "feature_space_class_distance_stats.csv", index=False, encoding="utf-8-sig")
    if nearest_frames:
        pd.concat(nearest_frames, ignore_index=True).to_csv(output_dir / "feature_space_nearest_conflicts.csv", index=False, encoding="utf-8-sig")
    write_markdown(summary_df, skipped_df, output_dir / "feature_space_report.md")

    print(f"[SUCCESS] Final feature-space comparison saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
