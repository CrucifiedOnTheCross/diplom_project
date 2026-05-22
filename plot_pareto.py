import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("all_experiments_summary.csv", sep=";")

df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
df["ece_after"] = pd.to_numeric(df["ece_after"], errors="coerce")

df = df.dropna(subset=["MCC", "ece_after"])

# =====================
# Pareto frontier
# =====================
df_sorted = df.sort_values(["ece_after", "MCC"], ascending=[True, False])

pareto = []
best_mcc = -1

for _, row in df_sorted.iterrows():
    if row["MCC"] > best_mcc:
        pareto.append(row)
        best_mcc = row["MCC"]

pareto_df = pd.DataFrame(pareto).reset_index(drop=True)
pareto_df["id"] = pareto_df.index + 1

# =====================
# Plot
# =====================
plt.figure(figsize=(10, 7))

# все модели (бледные)
plt.scatter(df["ece_after"], df["MCC"], alpha=0.25)

# Pareto (жирные)
plt.scatter(pareto_df["ece_after"], pareto_df["MCC"])

# линия Pareto
plt.plot(pareto_df["ece_after"], pareto_df["MCC"])

# подписи ТОЛЬКО для Pareto
for _, row in pareto_df.iterrows():
    plt.annotate(
        f"{int(row['id'])}",
        (row["ece_after"], row["MCC"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=12,
        fontweight="bold"
    )

plt.xlabel("ECE (after calibration)")
plt.ylabel("MCC")
plt.title("Pareto Frontier (MCC vs ECE)")
plt.grid(True)

plt.savefig("pareto_diploma.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================
# Legend table
# =====================
legend = pareto_df[["id", "Experiment_Folder", "MCC", "ece_after"]]
legend.to_csv("pareto_legend.csv", index=False)

print("Saved:")
print(" - pareto_diploma.png")
print(" - pareto_legend.csv")