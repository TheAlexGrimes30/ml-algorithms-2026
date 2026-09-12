from pathlib import Path

import kagglehub
import pandas as pd


dataset_path = kagglehub.dataset_download(
    "kumargh/pimaindiansdiabetescsv",
    output_dir="data"
)

dataset_path = Path(dataset_path)

csv_path = next(dataset_path.glob("*.csv"))


columns = [
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


df = pd.read_csv(
    csv_path,
    names=columns,
    header=None
)

columns_to_impute = [
    "plas",
    "pres",
    "skin",
    "test",
    "mass"
]

for column in columns_to_impute:
	median = df.loc[
		df[column] != 0,
		column
	].median()

	df.loc[
        df[column] == 0,
        column
    ] = median

print(df.head())
print(df.info())
print(df.describe())
