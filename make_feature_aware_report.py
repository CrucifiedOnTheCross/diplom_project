#!/usr/bin/env python3
"""Create comparison tables and figures for feature-aware GAN-mix experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = ["Accuracy", "Balanced_Accuracy", "MCC", "F1_Macro", "PR_AUC"]
MINORITY_RECALL_COLUMNS = ["akiec_Recall", "bcc_Recall", "df_Recall", "mel_Recall", "vasc_Recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature-aware GAN-mix comparison report.")
    parser.add_argument("--summary", default="all_experiments_summary.csv")
    parser.add_argument("--baseline", required=True, help="Baseline Experiment_Folder")
    parser.add_argument("--additional-baselines", nargs="*", default=[], help="Extra baseline rows to include in the comparison table")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment_Folder values to compare")
    parser.add_argument("--output-dir", default="feature_aware_report")
    return parser.parse_args()


def read_summary(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        return pd.read_csv(path)


def numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def make_comparison(df: pd.DataFrame, baseline: str, additional_baselines: List[str], experiments: List[str]) -> pd.DataFrame:
    wanted = []
    for name in [baseline, *additional_baselines, *experiments]:
        if name not in wanted:
            wanted.append(name)
    subset = df[df["Experiment_Folder"].isin(wanted)].copy()
    subset["_order"] = subset["Experiment_Folder"].map({name: i for i, name in enumerate(wanted)})
    subset = subset.sort_values("_order").drop(columns=["_order"])

    columns = ["Experiment_Folder"] + [col for col in METRIC_COLUMNS + MINORITY_RECALL_COLUMNS if col in subset.columns]
    return numeric(subset[columns], columns[1:])


def make_delta(comparison: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base_rows = comparison[comparison["Experiment_Folder"] == baseline]
    if base_rows.empty:
        raise ValueError(f"Baseline not found in summary: {baseline}")
    base = base_rows.iloc[0]

    delta = comparison.copy()
    for col in comparison.columns:
        if col == "Experiment_Folder":
            continue
        delta[col] = pd.to_numeric(delta[col], errors="coerce") - float(base[col])
    return delta


def plot_metric(comparison: pd.DataFrame, metric: str, out_path: Path, title: str, ylabel: str) -> None:
    if metric not in comparison.columns:
        return

    plot_df = comparison.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["Experiment_Folder"], plot_df[metric])
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(0, max(1.0, float(plot_df[metric].max()) * 1.12))
    ax.tick_params(axis="x", labelrotation=25, labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_minority_recalls(comparison: pd.DataFrame, out_path: Path) -> None:
    recall_cols = [col for col in MINORITY_RECALL_COLUMNS if col in comparison.columns]
    if not recall_cols:
        return

    plot_df = comparison[["Experiment_Folder"] + recall_cols].copy()
    for col in recall_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(how="all", subset=recall_cols)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = range(len(plot_df))
    width = 0.15
    offsets = [width * (i - (len(recall_cols) - 1) / 2) for i in range(len(recall_cols))]

    for offset, col in zip(offsets, recall_cols):
        ax.bar([i + offset for i in x], plot_df[col], width=width, label=col.replace("_Recall", ""))

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["Experiment_Folder"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Recall by minority class", fontsize=16)
    ax.set_ylabel("Recall", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Class", ncols=min(5, len(recall_cols)), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(Path(args.summary))
    comparison = make_comparison(summary, args.baseline, args.additional_baselines, args.experiments)
    delta = make_delta(comparison, args.baseline)

    comparison.to_csv(output_dir / "feature_aware_summary.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(output_dir / "feature_aware_delta_summary.csv", index=False, encoding="utf-8-sig")
    for extra_baseline in args.additional_baselines:
        try:
            make_delta(comparison, extra_baseline).to_csv(
                output_dir / f"feature_aware_delta_vs_{extra_baseline}.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except ValueError:
            print(f"[WARN] Additional baseline not found in summary: {extra_baseline}")

    plot_metric(comparison, "Balanced_Accuracy", fig_dir / "balanced_accuracy_by_experiment.png", "Balanced Accuracy by experiment", "Balanced Accuracy")
    plot_metric(comparison, "MCC", fig_dir / "mcc_by_experiment.png", "MCC by experiment", "MCC")
    plot_metric(comparison, "F1_Macro", fig_dir / "macro_f1_by_experiment.png", "Macro F1 by experiment", "Macro F1")
    plot_minority_recalls(comparison, fig_dir / "minority_class_recall.png")

    print(f"[SUCCESS] Feature-aware report saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
