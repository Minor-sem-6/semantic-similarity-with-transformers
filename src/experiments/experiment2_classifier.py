import os
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

from src.embeddings.sbert_embedder import SBERTEmbedder
from src.embeddings.pythia_embedder import PythiaEmbedder
from src.embeddings.t5_embedder import T5Embedder


def create_features(embedder, ref_texts, stu_texts):

    ref_emb = embedder.encode(ref_texts)
    stu_emb = embedder.encode(stu_texts)

    features = []

    for r, s in zip(ref_emb, stu_emb):
        cos_sim = cosine_similarity([r], [s])[0][0]
        diff = np.abs(r - s)
        prod = r * s 

        feature_vector = np.concatenate(([cos_sim], diff, prod))
        features.append(feature_vector)

    return np.array(features)


def compute_qwk(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)
    y_true = np.array(y_true).astype(int)
    return cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")


def train_and_evaluate(X, y):

    indices = np.arange(len(y))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pearson_corr, _ = pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    qwk = compute_qwk(y_test, y_pred)

    return y_test, y_pred, idx_test, pearson_corr, rmse, mae, qwk


def run_experiment(dataset_path, embedder):

    df = pd.read_csv(dataset_path)

    ref = df["reference_answer"].tolist()
    stu = df["student_answer"].tolist()
    scores = df["score"].tolist()

    X = create_features(embedder, ref, stu)
    y = np.array(scores)

    y_test, y_pred, idx_test, corr, rmse, mae, qwk = train_and_evaluate(X, y)

    print(f"Pearson: {corr:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"QWK: {qwk:.4f}")

    return df, idx_test, y_pred, corr, rmse, mae, qwk


if __name__ == "__main__":

    datasets = {
        "mohler": "Data/processed/mohler/mohler_processed.csv",
        "scientsbank": "Data/processed/scientsbank/scientsbank_processed.csv",
        "beetle": "Data/processed/beetle/beetle_processed.csv"
    }

    base_dir = "results/experiment2"
    pred_dir = os.path.join(base_dir, "predictions")

    os.makedirs(pred_dir, exist_ok=True)

    embedders = {
        "sbert": SBERTEmbedder(),
        "pythia": PythiaEmbedder(),
        "t5": T5Embedder()
    }

    summary = []

    for model_name, embedder in embedders.items():
        for name, path in datasets.items():

            print("\n==============================")
            print("Dataset:", name)
            print("==============================")

            df, idx_test, y_pred, corr, rmse, mae, qwk = run_experiment(path, embedder)

            df["predicted_score"] = np.nan
            df.loc[idx_test, "predicted_score"] = y_pred

            # ✅ Save predictions inside /predictions
            pred_path = os.path.join(
                pred_dir, f"{name}_{model_name}_classifier.csv"
            )
            df.to_csv(pred_path, index=False)

            # ✅ Summary row
            summary.append({
                "dataset": name,
                "model": f"{model_name}_classifier",
                "pearson": corr,
                "rmse": rmse,
                "mae": mae,
                "qwk": qwk
            })

    # ✅ Save metrics_summary OUTSIDE predictions
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_dir, "metrics_summary.csv"), index=False)

    print("\n✅ Experiment 2 completed successfully!")