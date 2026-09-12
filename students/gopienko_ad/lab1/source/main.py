from dataset import (
    load_dataset,
    analyze_dataset,
    impute_zero_values,
)


def main():
    df = load_dataset()

    print("Before preprocessing:")
    analyze_dataset(df)

    df = impute_zero_values(df)

    print("\nAfter preprocessing:")
    analyze_dataset(df)


if __name__ == "__main__":
    main()