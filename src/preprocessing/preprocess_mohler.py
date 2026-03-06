import pandas as pd
from text_cleaning import clean_text


def preprocess_mohler(input_path, output_path):

    df = pd.read_csv(input_path)

    # keep required columns
    df = df[["desired_answer", "student_answer", "score_avg"]]

    # rename columns
    df = df.rename(columns={
        "desired_answer": "reference_answer",
        "score_avg": "score"
    })

    # clean text
    df["reference_answer"] = df["reference_answer"].apply(clean_text)
    df["student_answer"] = df["student_answer"].apply(clean_text)

    # remove empty rows
    df = df.dropna()

    # save processed dataset
    df.to_csv(output_path, index=False)

    print("Mohler preprocessing completed.")


if __name__ == "__main__":

    input_file = "Data/raw/mohler/mohler_dataset_edited.csv"
    output_file = "Data/processed/mohler/mohler_processed.csv"

    preprocess_mohler(input_file, output_file)