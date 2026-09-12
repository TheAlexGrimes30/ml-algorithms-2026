from dataset import (
    load_dataset,
    analyze_dataset,
    impute_zero_values,
)
from students.gopienko_ad.lab1.source.dataset import StandardScaler, train_val_test_split


def main():
    df = load_dataset()
    df = impute_zero_values(df)

    analyze_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df,
        test_size=0.15,
        val_size=0.15
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.transform(X_test)


if __name__ == "__main__":
    main()