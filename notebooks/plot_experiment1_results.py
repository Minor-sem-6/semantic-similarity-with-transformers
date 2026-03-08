import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load metrics
df = pd.read_csv("results/experiment1/metrics_summary.csv")

# ---------- Pearson Plot ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="pearson", hue="model")

plt.title("Experiment 1: Pearson Correlation Comparison")
plt.xlabel("Dataset")
plt.ylabel("Pearson Correlation")

plt.tight_layout()
plt.savefig("results/experiment1/pearson_comparison.png")
plt.close()


# ---------- RMSE Plot ----------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="dataset", y="rmse", hue="model")

plt.title("Experiment 1: RMSE Comparison")
plt.xlabel("Dataset")
plt.ylabel("RMSE (Lower is Better)")

plt.tight_layout()
plt.savefig("results/experiment1/rmse_comparison.png")
plt.close()

print("Graphs saved successfully in results/experiment1/")

# ---------- Pearson Heatmap ----------
plt.figure(figsize=(6,4))

heatmap_data = df.pivot(index="dataset", columns="model", values="pearson")

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f"
)

plt.title("Experiment 1: Pearson Correlation Heatmap")
plt.xlabel("Model")
plt.ylabel("Dataset")

plt.tight_layout()
plt.savefig("results/experiment1/pearson_heatmap.png")
plt.close()