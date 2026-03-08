import os
import pandas as pd
from src.evaluation.evaluation_metrics import compute_metrics, save_metrics

# dataset score ranges
max_scores = {
    "mohler": 5,
    "scientsbank": 4,
    "beetle": 4
}

results_dir = "results/experiment1"

for file in os.listdir(results_dir):

    if file.endswith("_similarity.csv"):

        path = os.path.join(results_dir, file)

        df = pd.read_csv(path)

        dataset = file.split("_")[0]
        model = file.split("_")[1]

        scores = df["score"]
        similarity = df["similarity"]

        pearson, rmse, mae, qwk = compute_metrics(
            scores, similarity, max_scores[dataset]
        )

        print("\nEvaluating:", dataset, model)
        print("Pearson:", pearson)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("QWK:", qwk)

        save_metrics(dataset, model, pearson, rmse, mae, qwk)