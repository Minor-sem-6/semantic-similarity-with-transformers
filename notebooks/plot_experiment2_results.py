import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "results/experiment2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(f"{OUTPUT_DIR}/metrics_summary.csv")

palette = {
    "sbert_classifier": "#F44336",
    "pythia_classifier": "#2196F3",
    "t5_classifier": "#4CAF50"
}

sns.set(style="whitegrid")

# -------- Pearson --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="pearson", hue="model", palette=palette)
plt.title("Experiment 2: Pearson Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pearson_comparison.png")
plt.close()

# -------- RMSE --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="rmse", hue="model", palette=palette)
plt.title("Experiment 2: RMSE Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rmse_comparison.png")
plt.close()

# -------- MAE --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="mae", hue="model", palette=palette)
plt.title("Experiment 2: MAE Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mae_comparison.png")
plt.close()

# -------- QWK --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="qwk", hue="model", palette=palette)
plt.title("Experiment 2: QWK Comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/qwk_comparison.png")
plt.close()

# -------- Pearson Heatmap --------
heatmap_pearson = df.pivot(index="dataset", columns="model", values="pearson")
plt.figure(figsize=(6,4))
sns.heatmap(heatmap_pearson, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Experiment 2: Pearson Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pearson_heatmap.png")
plt.close()

# -------- RMSE Heatmap --------
heatmap_rmse = df.pivot(index="dataset", columns="model", values="rmse")
plt.figure(figsize=(6,4))
sns.heatmap(heatmap_rmse, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Experiment 2: RMSE Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rmse_heatmap.png")
plt.close()

# -------- QWK Heatmap --------
heatmap_qwk = df.pivot(index="dataset", columns="model", values="qwk")
plt.figure(figsize=(6,4))
sns.heatmap(heatmap_qwk, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Experiment 2: QWK Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/qwk_heatmap.png")
plt.close()

# -------- MAE Heatmap --------
heatmap_mae = df.pivot(index="dataset", columns="model", values="mae")

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_mae, annot=True, cmap="YlOrBr", fmt=".2f")

plt.title("Experiment 2: MAE Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mae_heatmap.png")
plt.close()


print("✅ Experiment 2 plots generated!")