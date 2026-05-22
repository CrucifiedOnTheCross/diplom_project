import pandas as pd
import matplotlib.pyplot as plt

INPUT = "all_experiments_summary.csv"

df = pd.read_csv(INPUT, sep=";")

required = [
    "Experiment_Folder",
    "MCC",
    "ece_after",
    "malignant_threshold_before",
    "malignant_threshold_after",
    "malignant_threshold_delta",
    "malignant_sensitivity_before",
    "malignant_sensitivity_after",
    "malignant_specificity_before",
    "malignant_specificity_after",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        "❌ В summary отсутствуют колонки:\n" + "\n".join(missing) +
        "\n\nСначала пересобери CSV через update_experiment_summary.py"
    )

# =====================
# Приведение типов
# =====================
numeric_cols = [
    "MCC",
    "ece_after",
    "malignant_threshold_before",
    "malignant_threshold_after",
    "malignant_threshold_delta",
    "malignant_sensitivity_before",
    "malignant_sensitivity_after",
    "malignant_specificity_before",
    "malignant_specificity_after",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["MCC", "ece_after"]).copy()

# =====================
# 📊 1. ГРАФИК
# =====================
def plot_threshold_shift():
    plot_df = df.dropna(subset=["malignant_threshold_delta"]).copy()

    label_names = set()
    label_names.add(plot_df.loc[plot_df["malignant_threshold_delta"].idxmax(), "Experiment_Folder"])
    label_names.add(plot_df.loc[plot_df["malignant_threshold_delta"].idxmin(), "Experiment_Folder"])
    label_names.add(plot_df.loc[plot_df["ece_after"].idxmin(), "Experiment_Folder"])
    label_names.add(plot_df.loc[plot_df["MCC"].idxmax(), "Experiment_Folder"])

    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df["ece_after"], plot_df["malignant_threshold_delta"], alpha=0.35)

    top = plot_df[plot_df["Experiment_Folder"].isin(label_names)]

    for _, row in top.iterrows():
        plt.annotate(
            row["Experiment_Folder"],
            (row["ece_after"], row["malignant_threshold_delta"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("ECE после калибровки")
    plt.ylabel("Сдвиг оптимального порога")
    plt.title("Сдвиг threshold vs ECE (malignant)")
    plt.grid(True)

    plt.savefig("threshold_shift_vs_ece.png", dpi=300, bbox_inches="tight")
    plt.close()

# =====================
# 📊 2. ТАБЛИЦА (КЛЮЧЕВОЕ)
# =====================
def build_clinical_table():
    table = df[[
        "Experiment_Folder",
        "MCC",
        "ece_after",

        "malignant_threshold_before",
        "malignant_threshold_after",
        "malignant_threshold_delta",

        "malignant_sensitivity_before",
        "malignant_sensitivity_after",

        "malignant_specificity_before",
        "malignant_specificity_after",
    ]].copy()

    # Дельты
    table["sensitivity_delta"] = (
        table["malignant_sensitivity_after"] -
        table["malignant_sensitivity_before"]
    )

    table["specificity_delta"] = (
        table["malignant_specificity_after"] -
        table["malignant_specificity_before"]
    )

    # Сортировка: лучшие модели сверху
    table = table.sort_values(
        ["MCC", "ece_after"],
        ascending=[False, True]
    )

    table.to_csv("threshold_clinical_summary.csv", index=False)

    return table

# =====================
# 📊 3. ТОП МОДЕЛИ
# =====================
def print_top_models(table):
    print("\n🔥 ТОП-5 моделей (MCC + calibration):\n")

    top = table.head(5)

    for _, row in top.iterrows():
        print(f"{row['Experiment_Folder']}")
        print(f"  MCC: {row['MCC']:.4f} | ECE: {row['ece_after']:.4f}")
        print(f"  Threshold: {row['malignant_threshold_before']:.3f} → {row['malignant_threshold_after']:.3f}")
        print(f"  Sensitivity: {row['malignant_sensitivity_before']:.3f} → {row['malignant_sensitivity_after']:.3f}")
        print(f"  Specificity: {row['malignant_specificity_before']:.3f} → {row['malignant_specificity_after']:.3f}")
        print()

# =====================
# RUN
# =====================
plot_threshold_shift()
table = build_clinical_table()
print_top_models(table)

print("\n✅ Saved:")
print(" - threshold_shift_vs_ece.png")
print(" - threshold_clinical_summary.csv")