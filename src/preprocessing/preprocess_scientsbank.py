import pandas as pd
import os
from text_cleaning import clean_text


def preprocess_scientsbank(input_path, output_path):

    df = pd.read_csv(input_path)

    # keep required columns
    df = df[["reference_answer", "student_answer", "label"]]

    # reverse scoring
    df["score"] = 4 - df["label"]

    # drop old label column
    df = df.drop(columns=["label"])

    # clean text
    df["reference_answer"] = df["reference_answer"].apply(clean_text)
    df["student_answer"] = df["student_answer"].apply(clean_text)

    # remove empty rows
    df = df.dropna()

    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save processed dataset
    df.to_csv(output_path, index=False)

    print("SciEntsBank preprocessing completed.")


if __name__ == "__main__":

    input_file = "Data/raw/scientsbank/scientsbank_raw.csv"
    output_file = "Data/processed/scientsbank/scientsbank_processed.csv"

    preprocess_scientsbank(input_file, output_file)