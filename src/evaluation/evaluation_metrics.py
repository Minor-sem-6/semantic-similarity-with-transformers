import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import os


def normalize_similarity(similarity):
    """
    Convert cosine similarity from [-1,1] → [0,1]
    """
    return (similarity + 1) / 2


def convert_to_score(normalized_similarity, max_score):
    """
    Scale normalized similarity to dataset score range
    """
    return normalized_similarity * max_score


def compute_metrics(true_scores, similarity, max_score):

    similarity = np.array(similarity)
    true_scores = np.array(true_scores)

    # normalize similarity
    norm_sim = normalize_similarity(similarity)

    # predicted continuous score
    pred_scores = convert_to_score(norm_sim, max_score)

    # rounded score for QWK
    pred_scores_rounded = np.round(pred_scores)
    # clip predictions to valid score range
    pred_scores_rounded = np.clip(pred_scores_rounded, 0, max_score)
    # convert to integer
    pred_scores_rounded = pred_scores_rounded.astype(int)
    true_scores = true_scores.astype(int)

    # metrics
    pearson_corr, _ = pearsonr(similarity, true_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
    mae = mean_absolute_error(true_scores, pred_scores)

    qwk = cohen_kappa_score(
        true_scores,
        pred_scores_rounded,
        weights="quadratic"
    )

    return pearson_corr, rmse, mae, qwk


def save_metrics(dataset, model, pearson, rmse, mae, qwk):

    file_path = "results/experiment1/metrics_summary.csv"

    row = {
        "dataset": dataset,
        "model": model,
        "pearson": pearson,
        "rmse": rmse,
        "mae": mae,
        "qwk": qwk
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(file_path, index=False)

    print("\nMetrics saved to metrics_summary.csv")