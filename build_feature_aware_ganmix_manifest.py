#!/usr/bin/env python3
"""Build a feature-aware GAN-mix manifest from real and synthetic embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select synthetic images by SupCon embedding geometry.")
    parser.add_argument("--real-train-embeddings", required=True)
    parser.add_argument("--real-train-metadata", required=True)
    parser.add_argument("--synthetic-embeddings", required=True)
    parser.add_argument("--synthetic-metadata", required=True)
    parser.add_argument("--selection-mode", required=True, choices=["random", "core", "diverse_core", "confident_core"])
    parser.add_argument("--synthetic-ratio", type=float, required=True)
    parser.add_argument("--require-correct-pred", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-own-quantile", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=-float("inf"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = {"image_path", "label"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    meta["label"] = meta["label"].astype(str)
    return meta


def compute_real_geometry(embeddings: np.ndarray, meta: pd.DataFrame):
    centroids: Dict[str, np.ndarray] = {}
    radii: Dict[str, float] = {}
    real_counts: Dict[str, int] = {}

    for cls in CLASS_NAMES:
        mask = meta["label"].to_numpy() == cls
        if not np.any(mask):
            continue
        cls_emb = embeddings[mask]
        centroid = cls_emb.mean(axis=0)
        distances = np.linalg.norm(cls_emb - centroid, axis=1)
        centroids[cls] = centroid
        radii[cls] = distances
        real_counts[cls] = int(mask.sum())

    return centroids, radii, real_counts


def annotate_synthetic(synthetic_embeddings: np.ndarray, synthetic_meta: pd.DataFrame, centroids: Dict[str, np.ndarray]) -> pd.DataFrame:
    classes = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[cls] for cls in classes])
    distances = pairwise_distances(synthetic_embeddings, centroid_matrix, metric="euclidean")

    rows = []
    labels = synthetic_meta["label"].astype(str).to_numpy()
    for idx, row in synthetic_meta.reset_index(drop=True).iterrows():
        order = np.argsort(distances[idx])
        nearest_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else nearest_idx
        label = labels[idx]
        own_distance = float(distances[idx, classes.index(label)]) if label in classes else np.nan
        nearest_distance = float(distances[idx, nearest_idx])
        second_distance = float(distances[idx, second_idx])

        out = row.to_dict()
        out.update(
            {
                "source": "synthetic",
                "distance_to_own_centroid": own_distance,
                "nearest_centroid_class": classes[nearest_idx],
                "nearest_centroid_distance": nearest_distance,
                "second_nearest_class": classes[second_idx],
                "second_nearest_distance": second_distance,
                "margin_to_nearest_other": float(second_distance - own_distance) if np.isfinite(own_distance) else np.nan,
            }
        )
        rows.append(out)

    return pd.DataFrame(rows)


def farthest_point_select(embeddings: np.ndarray, candidate_indices: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(candidate_indices) <= k:
        return candidate_indices

    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(candidate_indices))]
    remaining = set(int(i) for i in candidate_indices)
    remaining.remove(selected[0])

    candidate_emb = embeddings[candidate_indices]
    local_to_global = {local: int(global_idx) for local, global_idx in enumerate(candidate_indices)}
    global_to_local = {int(global_idx): local for local, global_idx in local_to_global.items()}
    min_dist = np.full(len(candidate_indices), np.inf, dtype=np.float64)

    while len(selected) < k and remaining:
        last_local = global_to_local[selected[-1]]
        dist = np.linalg.norm(candidate_emb - candidate_emb[last_local], axis=1)
        min_dist = np.minimum(min_dist, dist)
        for chosen in selected:
            min_dist[global_to_local[chosen]] = -np.inf

        next_local = int(np.argmax(min_dist))
        next_global = local_to_global[next_local]
        selected.append(next_global)
        remaining.remove(next_global)

    return np.array(selected, dtype=int)


def select_rows(
    mode: str,
    ratio: float,
    synthetic_embeddings: np.ndarray,
    synthetic_annotated: pd.DataFrame,
    radii: Dict[str, np.ndarray],
    real_counts: Dict[str, int],
    seed: int,
    require_correct_pred: bool,
    min_confidence: float,
    max_own_quantile: float,
    min_margin: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected_indices: List[int] = []

    for cls in CLASS_NAMES:
        if cls not in real_counts:
            continue
        target = int(round(real_counts[cls] * ratio))
        if target <= 0:
            continue

        cls_mask = synthetic_annotated["label"].to_numpy() == cls
        cls_indices = np.flatnonzero(cls_mask)
        if len(cls_indices) == 0:
            print(f"[WARN] No synthetic candidates for {cls}")
            continue

        if mode == "random":
            pool = cls_indices
        else:
            own = synthetic_annotated["distance_to_own_centroid"].to_numpy()
            nearest = synthetic_annotated["nearest_centroid_class"].to_numpy()
            radius = float(np.quantile(radii[cls], max_own_quantile)) if cls in radii else np.inf
            core_mask = cls_mask & (own <= radius) & (nearest == cls)

            if mode == "confident_core":
                if require_correct_pred:
                    if "predicted_label" not in synthetic_annotated.columns:
                        raise ValueError("--require-correct-pred needs predicted_label in synthetic metadata")
                    pred = synthetic_annotated["predicted_label"].astype(str).to_numpy()
                    core_mask = core_mask & (pred == cls)

                if min_confidence > 0:
                    if "confidence" not in synthetic_annotated.columns:
                        raise ValueError("--min-confidence needs confidence in synthetic metadata")
                    conf = pd.to_numeric(synthetic_annotated["confidence"], errors="coerce").fillna(-np.inf).to_numpy()
                    core_mask = core_mask & (conf >= min_confidence)

                margin = synthetic_annotated["margin_to_nearest_other"].to_numpy()
                core_mask = core_mask & (margin > min_margin)

            pool = np.flatnonzero(core_mask)
            if len(pool) == 0:
                print(f"[WARN] No {mode} candidates for {cls}")
                continue

        if mode in {"diverse_core", "confident_core"}:
            chosen = farthest_point_select(synthetic_embeddings, pool, target, seed + CLASS_NAMES.index(cls))
        else:
            chosen = rng.choice(pool, size=min(target, len(pool)), replace=False)

        if len(chosen) < target:
            print(f"[WARN] {cls}: selected {len(chosen)} of requested {target}")
        selected_indices.extend(int(i) for i in chosen)

    selected = synthetic_annotated.iloc[selected_indices].copy()
    selected["selection_mode"] = mode
    selected["synthetic_ratio"] = ratio
    return selected.sort_values(["label", "image_path"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    real_embeddings = np.load(args.real_train_embeddings)
    synthetic_embeddings = np.load(args.synthetic_embeddings)
    real_meta = load_metadata(Path(args.real_train_metadata))
    synthetic_meta = load_metadata(Path(args.synthetic_metadata))

    if len(real_embeddings) != len(real_meta):
        raise ValueError("Real embeddings and metadata have different lengths")
    if len(synthetic_embeddings) != len(synthetic_meta):
        raise ValueError("Synthetic embeddings and metadata have different lengths")

    centroids, radii, real_counts = compute_real_geometry(real_embeddings, real_meta)
    synthetic_annotated = annotate_synthetic(synthetic_embeddings, synthetic_meta, centroids)
    selected = select_rows(
        mode=args.selection_mode,
        ratio=args.synthetic_ratio,
        synthetic_embeddings=synthetic_embeddings,
        synthetic_annotated=synthetic_annotated,
        radii=radii,
        real_counts=real_counts,
        seed=args.random_state,
        require_correct_pred=args.require_correct_pred,
        min_confidence=args.min_confidence,
        max_own_quantile=args.max_own_quantile,
        min_margin=args.min_margin,
    )

    fieldnames = [
        "image_path",
        "label",
        "source",
        "selection_mode",
        "synthetic_ratio",
        "predicted_label",
        "confidence",
        "distance_to_own_centroid",
        "nearest_centroid_class",
        "nearest_centroid_distance",
        "second_nearest_class",
        "second_nearest_distance",
        "margin_to_nearest_other",
    ]
    for column in fieldnames:
        if column not in selected.columns:
            selected[column] = ""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[fieldnames].to_csv(output, index=False)

    summary = selected.groupby("label").size().reset_index(name="selected_count")
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(f"[SUCCESS] Wrote {len(selected)} selected synthetic rows to {output.resolve()}")


if __name__ == "__main__":
    main()
