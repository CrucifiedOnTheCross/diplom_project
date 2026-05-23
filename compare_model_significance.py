#!/usr/bin/env python3
"""Paired statistical significance tests for two model prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
PROB_COLS = [f"prob_{cls}" for cls in CLASS_NAMES]
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "pr_auc_macro",
    "roc_auc_macro",
    "mel_recall",
    "bcc_recall",
    "akiec_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two models on the same test predictions.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "true_label", "predicted_label", *PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.sort_values("image_path").reset_index(drop=True)
    return df


def validate_pair(a: pd.DataFrame, b: pd.DataFrame) -> None:
    if len(a) != len(b):
        raise ValueError(f"Different row counts: {len(a)} vs {len(b)}")
    if not a["image_path"].equals(b["image_path"]):
        raise ValueError("image_path columns do not match after sorting")
    if not a["true_label"].equals(b["true_label"]):
        raise ValueError("true_label columns do not match after sorting")


def labels_to_int(labels: pd.Series) -> np.ndarray:
    lookup = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    return labels.map(lookup).to_numpy(dtype=int)


def predictions_to_int(labels: pd.Series) -> np.ndarray:
    lookup = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    return labels.map(lookup).to_numpy(dtype=int)


def metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "pr_auc_macro":
        y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
        return float(average_precision_score(y_bin, probs, average="macro"))
    if metric == "roc_auc_macro":
        y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
        return float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    if metric.endswith("_recall"):
        cls = metric.replace("_recall", "")
        cls_idx = CLASS_NAMES.index(cls)
        return float(recall_score((y_true == cls_idx).astype(int), (y_pred == cls_idx).astype(int), zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = []
    for cls_idx in np.unique(y_true):
        idx = np.flatnonzero(y_true == cls_idx)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def bootstrap_ci(
    metric: str,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    prob_a: np.ndarray,
    pred_b: np.ndarray,
    prob_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = stratified_bootstrap_indices(y_true, rng)
        va = metric_value(metric, y_true[idx], pred_a[idx], prob_a[idx])
        vb = metric_value(metric, y_true[idx], pred_b[idx], prob_b[idx])
        deltas[i] = vb - va
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def permutation_p_value(
    metric: str,
    observed_delta: float,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    prob_a: np.ndarray,
    pred_b: np.ndarray,
    prob_b: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    extreme = 0
    n = len(y_true)
    for _ in range(n_permutations):
        swap = rng.random(n) < 0.5
        perm_pred_a = pred_a.copy()
        perm_pred_b = pred_b.copy()
        perm_prob_a = prob_a.copy()
        perm_prob_b = prob_b.copy()
        perm_pred_a[swap], perm_pred_b[swap] = pred_b[swap], pred_a[swap]
        perm_prob_a[swap], perm_prob_b[swap] = prob_b[swap], prob_a[swap]
        delta = metric_value(metric, y_true, perm_pred_b, perm_prob_b) - metric_value(metric, y_true, perm_pred_a, perm_prob_a)
        if abs(delta) >= abs(observed_delta):
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def mcnemar_result(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, object]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    both_correct = int(np.sum(a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        result = mcnemar(table, exact=True)
        pvalue = float(result.pvalue)
        statistic = float(result.statistic)
        status = "ok"
    except Exception as exc:
        pvalue = np.nan
        statistic = np.nan
        status = f"unavailable: {exc}"
    return {
        "mcnemar_both_correct": both_correct,
        "mcnemar_a_correct_b_wrong": a_correct_b_wrong,
        "mcnemar_a_wrong_b_correct": a_wrong_b_correct,
        "mcnemar_both_wrong": both_wrong,
        "mcnemar_statistic": statistic,
        "mcnemar_p_value": pvalue,
        "mcnemar_status": status,
    }


def compare(args: argparse.Namespace) -> pd.DataFrame:
    a_df = read_predictions(Path(args.a))
    b_df = read_predictions(Path(args.b))
    validate_pair(a_df, b_df)

    y_true = labels_to_int(a_df["true_label"])
    pred_a = predictions_to_int(a_df["predicted_label"])
    pred_b = predictions_to_int(b_df["predicted_label"])
    prob_a = a_df[PROB_COLS].to_numpy(dtype=np.float64)
    prob_b = b_df[PROB_COLS].to_numpy(dtype=np.float64)

    rng_boot = np.random.default_rng(args.seed)
    rng_perm = np.random.default_rng(args.seed + 1)
    mc = mcnemar_result(y_true, pred_a, pred_b)

    rows: List[dict] = []
    for metric in METRICS:
        value_a = metric_value(metric, y_true, pred_a, prob_a)
        value_b = metric_value(metric, y_true, pred_b, prob_b)
        delta = value_b - value_a
        ci_low, ci_high = bootstrap_ci(metric, y_true, pred_a, prob_a, pred_b, prob_b, args.n_bootstrap, rng_boot)
        p_value = permutation_p_value(metric, delta, y_true, pred_a, prob_a, pred_b, prob_b, args.n_permutations, rng_perm)
        row = {
            "model_a": args.name_a,
            "model_b": args.name_b,
            "metric": metric,
            "value_a": value_a,
            "value_b": value_b,
            "delta_b_minus_a": delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "permutation_p_value": p_value,
            "significant_p05": p_value < 0.05,
            "ci_excludes_zero": (ci_low > 0) or (ci_high < 0),
            "n": len(y_true),
        }
        if metric == "accuracy":
            row.update(mc)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = compare(args)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Saved significance comparison to: {out}")


if __name__ == "__main__":
    main()
