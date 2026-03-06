import argparse

from preprocess_mohler import preprocess_mohler
from preprocess_scientsbank import preprocess_scientsbank
from preprocess_beetle import preprocess_beetle


def main(dataset):

    if dataset == "mohler":
        preprocess_mohler(
            "data/raw/mohler/mohler_dataset_edited.csv",
            "data/processed/mohler/mohler_processed.csv"
        )

    elif dataset == "scientsbank":
        preprocess_scientsbank(
            "data/raw/scientsbank/scientsbank_raw.csv",
            "data/processed/scientsbank/scientsbank_processed.csv"
        )

    elif dataset == "beetle":
        preprocess_beetle(
            "data/raw/beetle/beetle_raw.csv",
            "data/processed/beetle/beetle_processed.csv"
        )

    else:
        print("Dataset not recognized")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(args.dataset)