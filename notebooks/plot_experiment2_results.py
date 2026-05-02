import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/experiment2/metrics_summary.csv")

# -------- Pearson Plot --------
plt.figure(figsize=(8,5))
palette = {
    "sbert_classifier": "#F44336",   
    "pythia_classifier": "#2196F3",  
    "t5_classifier": "#4CAF50" 
}
sns.barplot(data=df, x="dataset", y="pearson", hue="model")

plt.title("Experiment 2: Pearson Comparison")
plt.tight_layout()
plt.savefig("results/experiment2/pearson_comparison.png")
plt.close()

# -------- RMSE Plot --------
plt.figure(figsize=(8,5))
palette = {
    "sbert_classifier": "#F44336",   
    "pythia_classifier": "#2196F3",  
    "t5_classifier": "#4CAF50" 
}
sns.barplot(data=df, x="dataset", y="rmse", hue="model")

plt.title("Experiment 2: RMSE Comparison")
plt.tight_layout()
plt.savefig("results/experiment2/rmse_comparison.png")
plt.close()

# -------- Heatmap --------
heatmap_data = df.pivot(index="dataset", columns="model", values="pearson")

plt.figure(figsize=(6,4))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")

plt.title("Experiment 2: Pearson Heatmap")
plt.tight_layout()
plt.savefig("results/experiment2/pearson_heatmap.png")
plt.close()

print("✅ Experiment 2 plots generated!")