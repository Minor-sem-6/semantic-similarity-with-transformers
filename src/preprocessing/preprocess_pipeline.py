import argparse

from .preprocess_mohler import preprocess_mohler
from .preprocess_scientsbank import preprocess_scientsbank
from .preprocess_beetle import preprocess_beetle


def main(Dataset):

    if Dataset == "mohler":
        preprocess_mohler(
            "Data/raw/mohler/mohler_Dataset_edited.csv",
            "Data/processed/mohler/mohler_processed.csv"
        )

    elif Dataset == "scientsbank":
        preprocess_scientsbank(
            "Data/raw/scientsbank/scientsbank_raw.csv",
            "Data/processed/scientsbank/scientsbank_processed.csv"
        )

    elif Dataset == "beetle":
        preprocess_beetle(
            "Data/raw/beetle/beetle_raw.csv",
            "Data/processed/beetle/beetle_processed.csv"
        )

    elif Dataset == "all":
        preprocess_mohler(
            "Data/raw/mohler/mohler_Dataset_edited.csv",
            "Data/processed/mohler/mohler_processed.csv"
        )

        preprocess_scientsbank(
            "Data/raw/scientsbank/scientsbank_raw.csv",
            "Data/processed/scientsbank/scientsbank_processed.csv"
        )

        preprocess_beetle(
            "Data/raw/beetle/beetle_raw.csv",
            "Data/processed/beetle/beetle_processed.csv"
        )

    else:
        print("Dataset not recognized")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(args.dataset)