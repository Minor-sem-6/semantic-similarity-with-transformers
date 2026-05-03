import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/experiment1/metrics_summary.csv")

# -------- Model Order --------
model_order = ["sbert", "pythia", "t5"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

# -------- Dataset Order --------
dataset_order = ["mohler", "scientsbank", "beetle"]
df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)

# -------- Palette --------
palette = {
    "sbert": "#F44336",
    "pythia": "#2196F3",
    "t5": "#4CAF50"
}

sns.set(style="whitegrid")

# ---------- Pearson ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="pearson",
            hue="model", hue_order=model_order, palette=palette)
plt.title("Experiment 1: Pearson Comparison")
plt.tight_layout()
plt.savefig("results/experiment1/pearson_comparison.png")
plt.close()

# ---------- RMSE ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="rmse",
            hue="model", hue_order=model_order, palette=palette)
plt.title("Experiment 1: RMSE Comparison")
plt.tight_layout()
plt.savefig("results/experiment1/rmse_comparison.png")
plt.close()

# ---------- MAE ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="mae",
            hue="model", hue_order=model_order, palette=palette)
plt.title("Experiment 1: MAE Comparison")
plt.tight_layout()
plt.savefig("results/experiment1/mae_comparison.png")
plt.close()

# ---------- QWK ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="qwk",
            hue="model", hue_order=model_order, palette=palette)
plt.title("Experiment 1: QWK Comparison")
plt.tight_layout()
plt.savefig("results/experiment1/qwk_comparison.png")
plt.close()

# ---------- Pearson Heatmap ----------
heatmap_pearson = df.pivot(index="dataset", columns="model", values="pearson")
heatmap_pearson = heatmap_pearson[model_order]

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_pearson, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Experiment 1: Pearson Heatmap")
plt.tight_layout()
plt.savefig("results/experiment1/pearson_heatmap.png")
plt.close()

# ---------- RMSE Heatmap ----------
heatmap_rmse = df.pivot(index="dataset", columns="model", values="rmse")
heatmap_rmse = heatmap_rmse[model_order]

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_rmse, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Experiment 1: RMSE Heatmap")
plt.tight_layout()
plt.savefig("results/experiment1/rmse_heatmap.png")
plt.close()

# ---------- QWK Heatmap ----------
heatmap_qwk = df.pivot(index="dataset", columns="model", values="qwk")
heatmap_qwk = heatmap_qwk[model_order]

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_qwk, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Experiment 1: QWK Heatmap")
plt.tight_layout()
plt.savefig("results/experiment1/qwk_heatmap.png")
plt.close()

# ---------- MAE Heatmap ----------
heatmap_mae = df.pivot(index="dataset", columns="model", values="mae")
heatmap_mae = heatmap_mae[model_order]

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_mae, annot=True, cmap="YlOrBr", fmt=".2f")

plt.title("Experiment 1: MAE Heatmap")
plt.tight_layout()
plt.savefig("results/experiment1/mae_heatmap.png")
plt.close()

print("✅ Experiment 1 plots generated!")