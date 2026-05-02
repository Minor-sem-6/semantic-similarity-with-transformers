# src/experiments/plot_experiment3.py


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results/experiment3_finetune"

# ── Palette matching your experiment ──────────
palette = {
    "sbert_finetune":  "#F44336",
    "pythia_finetune": "#2196F3",
    "t5_finetune":     "#4CAF50",
}


def plot_experiment3():

    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found. Run experiment3 first.")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["pearson", "rmse"])

    # ── Pearson Plot ──────────────────────────
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="dataset", y="pearson",
                hue="model", palette=palette)
    plt.title("Experiment 3: Pearson Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pearson_comparison.png"))
    plt.close()

    # ── RMSE Plot ─────────────────────────────
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="dataset", y="rmse",
                hue="model", palette=palette)
    plt.title("Experiment 3: RMSE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rmse_comparison.png"))
    plt.close()

    # ── MAE Plot ──────────────────────────────
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="dataset", y="mae",
                hue="model", palette=palette)
    plt.title("Experiment 3: MAE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mae_comparison.png"))
    plt.close()

    # ── QWK Plot ──────────────────────────────
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="dataset", y="qwk",
                hue="model", palette=palette)
    plt.title("Experiment 3: QWK Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qwk_comparison.png"))
    plt.close()

    # ── Pearson Heatmap ───────────────────────
    heatmap_data = df.pivot(
        index="dataset", columns="model", values="pearson"
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Experiment 3: Pearson Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pearson_heatmap.png"))
    plt.close()

    # ── RMSE Heatmap ──────────────────────────
    heatmap_rmse = df.pivot(
        index="dataset", columns="model", values="rmse"
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_rmse, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Experiment 3: RMSE Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rmse_heatmap.png"))
    plt.close()

    print("✅ Experiment 3 plots generated!")


if __name__ == "__main__":
    plot_experiment3()