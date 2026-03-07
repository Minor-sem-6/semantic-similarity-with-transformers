import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from src.embeddings.pythia_embedder import PythiaEmbedder
from src.embeddings.sbert_embedder import SBERTEmbedder

# teammates will add
# from src.embeddings.t5_embedder import T5Embedder


def load_dataset(path):

    df = pd.read_csv(path)

    ref_answers = df["reference_answer"].tolist()
    stu_answers = df["student_answer"].tolist()
    scores = df["score"].tolist()

    return ref_answers, stu_answers, scores


def compute_similarity(embedder, ref_texts, stu_texts):

    ref_embeddings = embedder.encode(ref_texts)
    stu_embeddings = embedder.encode(stu_texts)

    similarities = cosine_similarity(ref_embeddings, stu_embeddings)

    similarities = similarities.diagonal()

    return similarities


def evaluate(similarities, scores):

    corr, _ = pearsonr(similarities, scores)

    return corr


def run_experiment(dataset_path, model_name):

    print(f"\nRunning similarity experiment: {model_name}")

    ref, stu, scores = load_dataset(dataset_path)

    if model_name == "pythia":
        embedder = PythiaEmbedder()

    elif model_name == "sbert":
        embedder = SBERTEmbedder()

    # teammates will add
    # elif model_name == "t5":
    #     embedder = T5Embedder()

    similarities = compute_similarity(embedder, ref, stu)

    correlation = evaluate(similarities, scores)

    print("Pearson correlation:", correlation)

    return similarities, correlation


if __name__ == "__main__":

    datasets = {
        "mohler": "Data/processed/mohler/mohler_processed.csv",
        "scientsbank": "Data/processed/scientsbank/scientsbank_processed.csv",
        "beetle": "Data/processed/beetle/beetle_processed.csv"
    }

    models = ["pythia", "sbert"]  # teammates will add "t5"

    for name, path in datasets.items():

        print("\nDataset:", name)
        for model in models:
            similarities, corr = run_experiment(path, model)
            df = pd.read_csv(path)
            df["similarity"] = similarities
            
            output_dir = "results/experiment1"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(f"{output_dir}/{name}_{model}_similarity.csv", index=False)