from pathlib import Path

import kagglehub
import pandas as pd
import numpy as np


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

def train_test_split(
        df: pd.DataFrame,
        target_column: str = "class",
        test_size: float = 0.2,
        random_state: int = 42
):
    X = df.drop(columns=[target_column]).to_numpy()
    y = df[target_column].to_numpy()

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    test_count = int(len(df) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

class StandardScaler:
    def __init__(self, eps: float = 1e-12):
        self.means = None
        self.stds = None 
        self.eps = eps 

    def fit(
            self,
            X: np.ndarray
    ) -> None:

        X = X.astype(float)

        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

        self.std = np.where(
            self.stds < self.eps,
            1.0,
            self.stds
        )

    def transform(self, X: np.ndarray) -> np.ndarray:

        if (self.means is None or self.stds is None):
            raise ValueError(
                "StandardScaler must be fitted before transform."
            )

        X = X.astype(float)

        X_scaled = (X - self.means) / self.stds

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
