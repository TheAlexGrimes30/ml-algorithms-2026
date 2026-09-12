from pathlib import Path

import kagglehub
import pandas as pd


COLUMNS = [
    "preg",
    "plas",
    "pres",
    "skin",
    "test",
    "mass",
    "pedi",
    "age",
    "class"
]


COLUMNS_TO_IMPUTE = [
    "plas",
    "pres",
    "skin",
    "test",
    "mass"
]


def load_dataset() -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download(
        "kumargh/pimaindiansdiabetescsv",
        output_dir="data"
    )

    dataset_path = Path(dataset_path)

    csv_path = next(
        dataset_path.glob("*.csv")
    )

    df = pd.read_csv(
        csv_path,
        names=COLUMNS,
        header=None
    )

    return df


def analyze_dataset(
        df: pd.DataFrame
) -> None:
    print("First rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nStatistics:")
    print(df.describe())

    print("\nZero values:")
    print(
        (df[COLUMNS_TO_IMPUTE] == 0).sum()
    )


def impute_zero_values(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    for column in COLUMNS_TO_IMPUTE:
        median = df.loc[
            df[column] != 0,
            column
        ].median()

        df.loc[
            df[column] == 0,
            column
        ] = median

    return df