# src/experiments/experiment3_finetune.py
"""
Experiment 3: Transformer (trainable) → Embeddings → Random Forest → Score
Run: python -m src.experiments.experiment3_finetune
"""

import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from src.finetuning.sbert_finetune  import train_sbert
from src.finetuning.pythia_finetune import train_pythia
from src.finetuning.t5_finetune     import train_t5

# ── Config ────────────────────────────────────
DATASETS = {
    "mohler":      "Data/processed/mohler/mohler_processed.csv",
    "scientsbank": "Data/processed/scientsbank/scientsbank_processed.csv",
    "beetle":      "Data/processed/beetle/beetle_processed.csv",
}

TRAINERS = {
    "sbert_finetune":  train_sbert,
    "pythia_finetune": train_pythia,
    "t5_finetune":     train_t5,
}

OUTPUT_DIR = "results/experiment3"
PRED_DIR   = os.path.join(OUTPUT_DIR, "predictions")
MODEL_DIR  = "saved_models/experiment3"


def run_experiment3():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    summary = []

    for model_name, train_fn in TRAINERS.items():
        for dataset_name, dataset_path in DATASETS.items():

            if not os.path.exists(dataset_path):
                print(f"  ⚠️  {dataset_path} not found — skipping")
                continue

            print("\n" + "=" * 60)
            print(f"  MODEL   : {model_name}")
            print(f"  DATASET : {dataset_name}")
            print(f"  PIPELINE: Transformer → Embeddings → RF → Score")
            print("=" * 60)

            save_path = os.path.join(
                MODEL_DIR, f"{model_name}_{dataset_name}.pt"
            )
            predictions_path = os.path.join(
                PRED_DIR, f"{dataset_name}_{model_name}.csv"
            )

            try:
                metrics = train_fn(
                    dataset_path     = dataset_path,
                    save_path        = save_path,
                    predictions_path = predictions_path,
                    epochs           = 5,
                )

                summary.append({
                    "model":   model_name,
                    "dataset": dataset_name,
                    **metrics
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                summary.append({
                    "model":   model_name,
                    "dataset": dataset_name,
                    "pearson": None,
                    "rmse":    None,
                    "mae":     None,
                    "qwk":     None,
                })

    # Save metrics CSV
    summary_df = pd.DataFrame(summary)
    out_csv    = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n✅ Metrics     → {out_csv}")
    print(f"✅ Predictions → {PRED_DIR}/")

    return summary_df


if __name__ == "__main__":
    run_experiment3()