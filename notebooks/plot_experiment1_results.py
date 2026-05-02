import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load metrics
df = pd.read_csv("results/experiment1/metrics_summary.csv")

# -------- Force correct model order --------
model_order = ["sbert", "pythia", "t5"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

# -------- Color palette --------
palette = {
    "sbert": "#F44336",   
    "pythia": "#2196F3",  
    "t5": "#4CAF50" 
}

sns.set(style="whitegrid")

# ---------- Pearson Plot ----------
plt.figure(figsize=(8,5))
sns.barplot(
    data=df, x="dataset", y="pearson",
    hue="model", hue_order=model_order, palette=palette
)

plt.title("Experiment 1: Pearson Correlation Comparison")
plt.xlabel("Dataset")
plt.ylabel("Pearson Correlation")

plt.tight_layout()
plt.savefig("results/experiment1/pearson_comparison.png")
plt.close()


# ---------- RMSE Plot ----------
plt.figure(figsize=(8,5))
sns.barplot(
    data=df, x="dataset", y="rmse",
    hue="model", hue_order=model_order, palette=palette
)

plt.title("Experiment 1: RMSE Comparison")
plt.xlabel("Dataset")
plt.ylabel("RMSE (Lower is Better)")

plt.tight_layout()
plt.savefig("results/experiment1/rmse_comparison.png")
plt.close()


# ---------- MAE Plot ----------
plt.figure(figsize=(8,5))
sns.barplot(
    data=df, x="dataset", y="mae",
    hue="model", hue_order=model_order, palette=palette
)

plt.title("Experiment 1: MAE Comparison")
plt.xlabel("Dataset")
plt.ylabel("MAE (Lower is Better)")

plt.tight_layout()
plt.savefig("results/experiment1/mae_comparison.png")
plt.close()


# ---------- Pearson Heatmap ----------
plt.figure(figsize=(6,4))

heatmap_data = df.pivot(index="dataset", columns="model", values="pearson")
heatmap_data = heatmap_data[model_order]  # enforce column order

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

print("✅ Graphs (Pearson, RMSE, MAE, Heatmap) saved in results/experiment1/")