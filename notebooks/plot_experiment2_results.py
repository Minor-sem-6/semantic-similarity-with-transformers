import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output folder exists
os.makedirs("results/experiment2", exist_ok=True)

# Load data
df = pd.read_csv("results/experiment2/metrics_summary.csv")

# -------- Common Style --------
palette = {
    "sbert_classifier": "#F44336",   # red
    "pythia_classifier": "#2196F3",  # blue
    "t5_classifier": "#4CAF50"       # green
}

sns.set(style="whitegrid")

# -------- Pearson Plot --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="pearson", hue="model", palette=palette)

plt.title("Experiment 2: Pearson Comparison")
plt.xlabel("Dataset")
plt.ylabel("Pearson Correlation")

plt.tight_layout()
plt.savefig("results/experiment2/pearson_comparison.png")
plt.close()


# -------- RMSE Plot --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="rmse", hue="model", palette=palette)

plt.title("Experiment 2: RMSE Comparison")
plt.xlabel("Dataset")
plt.ylabel("RMSE")

plt.tight_layout()
plt.savefig("results/experiment2/rmse_comparison.png")
plt.close()


# -------- MAE Plot --------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="mae", hue="model", palette=palette)

plt.title("Experiment 2: MAE Comparison")
plt.xlabel("Dataset")
plt.ylabel("MAE")

plt.tight_layout()
plt.savefig("results/experiment2/mae_comparison.png")
plt.close()


# -------- Heatmap (Pearson) --------
heatmap_data = df.pivot(index="dataset", columns="model", values="pearson")

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")

plt.title("Experiment 2: Pearson Heatmap")
plt.tight_layout()
plt.savefig("results/experiment2/pearson_heatmap.png")
plt.close()


print("✅ Experiment 2 plots (Pearson, RMSE, MAE, Heatmap) generated!")