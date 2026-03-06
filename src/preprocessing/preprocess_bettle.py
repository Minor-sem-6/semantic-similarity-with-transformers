import pandas as pd
from text_cleaning import clean_text


def preprocess_beetle(input_path, output_path):

    df = pd.read_csv(input_path)

    df = df[["reference_answer", "student_answer", "label"]]

    df["label"] = 4 - df["label"]

    df["reference_answer"] = df["reference_answer"].apply(clean_text)
    df["student_answer"] = df["student_answer"].apply(clean_text)

    df = df.dropna()

    df.to_csv(output_path, index=False)

    print("Beetle preprocessing completed.")


if __name__ == "__main__":

    input_file = "Data/raw/beetle/beetle_raw.csv"
    output_file = "Data/processed/beetle/beetle_processed.csv"

    preprocess_beetle(input_file, output_file)