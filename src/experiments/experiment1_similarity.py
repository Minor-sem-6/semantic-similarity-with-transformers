import os
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

from src.embeddings.pythia_embedder import PythiaEmbedder
from src.embeddings.sbert_embedder import SBERTEmbedder
from src.embeddings.t5_embedder import T5Embedder


def load_dataset(path):
    df = pd.read_csv(path)

    ref_answers = df["reference_answer"].tolist()
    stu_answers = df["student_answer"].tolist()
    scores = df["score"].tolist()

    return df, ref_answers, stu_answers, scores


def compute_similarity(embedder, ref_texts, stu_texts):
    ref_embeddings = embedder.encode(ref_texts)
    stu_embeddings = embedder.encode(stu_texts)

    similarities = cosine_similarity(ref_embeddings, stu_embeddings)
    return similarities.diagonal()


def scale_predictions(similarities, scores):
    """
    Scale cosine similarities to match score range
    """
    min_score = min(scores)
    max_score = max(scores)

    # cosine similarity is [-1, 1] → convert to [0,1]
    sims = (similarities + 1) / 2

    # scale to score range
    scaled = sims * (max_score - min_score) + min_score
    return scaled


def evaluate(similarities, scores):

    # Pearson
    pearson, _ = pearsonr(similarities, scores)

    # Scale predictions
    preds = scale_predictions(similarities, scores)

    # RMSE
    rmse = np.sqrt(mean_squared_error(scores, preds))

    # MAE
    mae = mean_absolute_error(scores, preds)

    # QWK (needs discrete values)
    preds_rounded = np.round(preds)
    scores_rounded = np.round(scores)

    qwk = cohen_kappa_score(scores_rounded, preds_rounded, weights="quadratic")

    return pearson, rmse, mae, qwk, preds


def get_embedder(model_name):
    if model_name == "pythia":
        return PythiaEmbedder()
    elif model_name == "sbert":
        return SBERTEmbedder()
    elif model_name == "t5":
        return T5Embedder()
    else:
        raise ValueError("Unknown model")


if __name__ == "__main__":

    datasets = {
        "mohler": "Data/processed/mohler/mohler_processed.csv",
        "scientsbank": "Data/processed/scientsbank/scientsbank_processed.csv",
        "beetle": "Data/processed/beetle/beetle_processed.csv"
    }

    models = ["pythia", "sbert", "t5"]

    base_dir = "results/experiment1"
    pred_dir = os.path.join(base_dir, "predictions")

    os.makedirs(pred_dir, exist_ok=True)

    summary_rows = []

    for dataset_name, path in datasets.items():

        print(f"\n📊 Dataset: {dataset_name}")

        df_original, ref, stu, scores = load_dataset(path)

        for model in models:

            print(f"Running model: {model}")

            embedder = get_embedder(model)

            similarities = compute_similarity(embedder, ref, stu)

            pearson, rmse, mae, qwk, preds = evaluate(similarities, scores)

            print(f"Pearson: {pearson:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | QWK: {qwk:.4f}")

            # -------- Save predictions --------
            df_pred = df_original.copy()
            df_pred["similarity"] = similarities
            df_pred["pred_score"] = preds

            pred_path = os.path.join(
                pred_dir, f"{dataset_name}_{model}.csv"
            )
            df_pred.to_csv(pred_path, index=False)

            # -------- Save summary --------
            summary_rows.append({
                "dataset": dataset_name,
                "model": model,
                "pearson": pearson,
                "rmse": rmse,
                "mae": mae,
                "qwk": qwk
            })

    # -------- Save metrics summary --------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(base_dir, "metrics_summary.csv"), index=False)

    print("\n✅ Experiment 1 completed with full metrics!")