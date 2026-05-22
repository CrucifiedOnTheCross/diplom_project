#!/usr/bin/env python3
"""Analyze SupCon embedding geometry for HAM10000 experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FOCUS_PAIRS = [("mel", "nv"), ("akiec", "bkl"), ("bcc", "bkl")]
FOCUS_CLASSES = ["df", "vasc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class geometry in exported embedding space.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def label_to_name(value) -> str:
    if isinstance(value, str) and value in CLASS_NAMES:
        return value
    try:
        return CLASS_NAMES[int(value)]
    except Exception:
        return str(value)


def load_metadata(path: Path, labels: np.ndarray) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "label" not in meta.columns:
        meta["label"] = [label_to_name(x) for x in labels]
    else:
        meta["label"] = meta["label"].map(label_to_name)
    return meta


def compute_centroids(embeddings: np.ndarray, label_names: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    classes = [cls for cls in CLASS_NAMES if np.any(label_names == cls)]
    centroids = []
    for cls in classes:
        centroids.append(embeddings[label_names == cls].mean(axis=0))
    return np.vstack(centroids), classes


def write_distance_reports(
    embeddings: np.ndarray,
    label_names: np.ndarray,
    centroids: np.ndarray,
    classes: List[str],
    output_dir: Path,
) -> None:
    centroid_lookup = {cls: centroids[i] for i, cls in enumerate(classes)}
    rows = []
    radii: Dict[str, float] = {}

    for cls in classes:
        cls_emb = embeddings[label_names == cls]
        distances = np.linalg.norm(cls_emb - centroid_lookup[cls], axis=1)
        radii[cls] = float(np.quantile(distances, 0.75))
        rows.append(
            {
                "class": cls,
                "n": int(len(distances)),
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "median_distance": float(np.median(distances)),
                "q75_distance": radii[cls],
                "q90_distance": float(np.quantile(distances, 0.90)),
                "max_distance": float(np.max(distances)),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "class_distance_stats.csv", index=False)

    centroid_dists = pairwise_distances(centroids, metric="euclidean")
    pd.DataFrame(centroid_dists, index=classes, columns=classes).to_csv(output_dir / "centroid_distances.csv")

    focus_rows = []
    for a, b in FOCUS_PAIRS:
        if a in classes and b in classes:
            focus_rows.append({"class_a": a, "class_b": b, "centroid_distance": centroid_dists[classes.index(a), classes.index(b)]})

    for cls in FOCUS_CLASSES:
        if cls in classes:
            idx = classes.index(cls)
            candidates = [(classes[j], centroid_dists[idx, j]) for j in range(len(classes)) if j != idx]
            nearest, distance = min(candidates, key=lambda item: item[1])
            focus_rows.append({"class_a": cls, "class_b": nearest, "centroid_distance": distance})

    nearest_rows = []
    for i, cls in enumerate(classes):
        candidates = [(classes[j], centroid_dists[i, j]) for j in range(len(classes)) if j != i]
        nearest, distance = min(candidates, key=lambda item: item[1])
        nearest_rows.append({"class": cls, "nearest_class": nearest, "centroid_distance": distance})

    pd.DataFrame(focus_rows).to_csv(output_dir / "focus_class_pairs.csv", index=False)
    pd.DataFrame(nearest_rows).to_csv(output_dir / "nearest_conflicting_classes.csv", index=False)


def write_silhouette(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    rows = []
    try:
        if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
            score = silhouette_score(embeddings, labels, metric="euclidean")
            rows.append({"metric": "silhouette_score", "value": float(score), "status": "ok"})
        else:
            rows.append({"metric": "silhouette_score", "value": "", "status": "not_enough_classes"})
    except Exception as exc:
        rows.append({"metric": "silhouette_score", "value": "", "status": f"error: {exc}"})

    pd.DataFrame(rows).to_csv(output_dir / "silhouette_summary.csv", index=False)


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


def plot_by_label(coords: np.ndarray, meta: pd.DataFrame, indices: np.ndarray, method: str, output_dir: Path) -> None:
    plot_meta = meta.iloc[indices].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in CLASS_NAMES:
        mask = plot_meta["label"].to_numpy() == cls
        if np.any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=cls)
    ax.set_title(f"{method.upper()} of SupCon embeddings by true label", fontsize=15)
    ax.set_xlabel(f"{method.upper()} 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Class", fontsize=9, title_fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{method}_by_true_label.png", dpi=300)
    plt.close(fig)


def plot_errors(coords: np.ndarray, meta: pd.DataFrame, indices: np.ndarray, method: str, output_dir: Path) -> None:
    if "predicted_label" not in meta.columns:
        return

    plot_meta = meta.iloc[indices].reset_index(drop=True)
    pred = plot_meta["predicted_label"].fillna("").astype(str).to_numpy()
    true = plot_meta["label"].astype(str).to_numpy()
    has_pred = pred != ""
    errors = has_pred & (pred != true)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[~errors, 0], coords[~errors, 1], s=14, alpha=0.25, label="correct/unknown")
    if np.any(errors):
        ax.scatter(coords[errors, 0], coords[errors, 1], s=28, alpha=0.9, label="prediction error")
    ax.set_title(f"{method.upper()} of SupCon embeddings with errors highlighted", fontsize=15)
    ax.set_xlabel(f"{method.upper()} 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{method}_errors_highlighted.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    labels = np.load(args.labels)
    label_names = np.array([label_to_name(x) for x in labels])
    meta = load_metadata(Path(args.metadata), labels)

    centroids, classes = compute_centroids(embeddings, label_names)
    np.save(output_dir / "class_centroids.npy", centroids)
    with (output_dir / "class_centroid_labels.txt").open("w", encoding="utf-8") as f:
        for cls in classes:
            f.write(cls + "\n")

    write_distance_reports(embeddings, label_names, centroids, classes, output_dir)
    write_silhouette(embeddings, labels, output_dir)

    coords, indices, method = compute_projection(embeddings, args.max_plot_points, args.random_state)
    plot_by_label(coords, meta, indices, method, output_dir)
    plot_errors(coords, meta, indices, method, output_dir)

    print(f"[SUCCESS] Embedding-space reports saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
