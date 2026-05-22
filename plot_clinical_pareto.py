#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CANDIDATES = [
    Path("all_experiments_summary.csv"),
    Path("science_folder") / "all_experiments_summary.csv",
]

def load_table():
    for path in INPUT_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path, sep=';')
                return df, path
            except Exception:
                pass
            try:
                df = pd.read_csv(path, sep=',')
                return df, path
            except Exception:
                pass
    raise FileNotFoundError("Не найден all_experiments_summary.csv")

def pareto_front_max_max(df, x_col, y_col):
    # maximize x and y
    work = df.dropna(subset=[x_col, y_col]).copy()
    work = work.sort_values([x_col, y_col], ascending=[False, False]).reset_index(drop=True)

    pareto_rows = []
    best_y = float("-inf")
    for _, row in work.iterrows():
        if row[y_col] > best_y:
            pareto_rows.append(row)
            best_y = row[y_col]

    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df = pareto_df.sort_values(x_col, ascending=True).reset_index(drop=True)
    pareto_df["id"] = pareto_df.index + 1
    return pareto_df

def make_plot(df, x_col, y_col, title, xlabel, ylabel, out_png, out_csv):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    pareto_df = pareto_front_max_max(plot_df, x_col, y_col)

    fig, ax = plt.subplots(figsize=(10, 7))

    # all models
    ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.25)

    # Pareto points
    ax.scatter(pareto_df[x_col], pareto_df[y_col], s=90)
    ax.plot(pareto_df[x_col], pareto_df[y_col], linewidth=2)

    # label only pareto models
    for _, row in pareto_df.iterrows():
        ax.annotate(
            f"{int(row['id'])}",
            xy=(row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    legend = pareto_df[["id", "Experiment_Folder", x_col, y_col]].copy()
    legend.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return pareto_df

def main():
    df, src = load_table()
    print(f"Using: {src}")

    required = [
        "Experiment_Folder",
        "MCC",
        "melanoma_sensitivity_after",
        "melanoma_specificity_after",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "В файле отсутствуют колонки: " + ", ".join(missing) +
            ". Сначала обнови update_experiment_summary.py и пересобери all_experiments_summary.csv"
        )

    for col in ["MCC", "melanoma_sensitivity_after", "melanoma_specificity_after"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    p1 = make_plot(
        df=df,
        x_col="melanoma_specificity_after",
        y_col="melanoma_sensitivity_after",
        title="Парето-оптимальные модели по melanoma sensitivity и specificity",
        xlabel="Melanoma Specificity после калибровки",
        ylabel="Melanoma Sensitivity после калибровки",
        out_png="pareto_melanoma_sens_spec_after.png",
        out_csv="pareto_melanoma_sens_spec_after_legend.csv",
    )

    p2 = make_plot(
        df=df,
        x_col="melanoma_sensitivity_after",
        y_col="MCC",
        title="Парето-оптимальные модели по MCC и melanoma sensitivity",
        xlabel="Melanoma Sensitivity после калибровки",
        ylabel="MCC",
        out_png="pareto_mcc_vs_melanoma_sens_after.png",
        out_csv="pareto_mcc_vs_melanoma_sens_after_legend.csv",
    )

    print("\nSaved:")
    print(" - pareto_melanoma_sens_spec_after.png")
    print(" - pareto_melanoma_sens_spec_after_legend.csv")
    print(" - pareto_mcc_vs_melanoma_sens_after.png")
    print(" - pareto_mcc_vs_melanoma_sens_after_legend.csv")

    print("\nPareto: melanoma sensitivity vs specificity")
    print(p1[["id", "Experiment_Folder", "melanoma_specificity_after", "melanoma_sensitivity_after"]].to_string(index=False))

    print("\nPareto: MCC vs melanoma sensitivity")
    print(p2[["id", "Experiment_Folder", "melanoma_sensitivity_after", "MCC"]].to_string(index=False))

if __name__ == "__main__":
    main()
