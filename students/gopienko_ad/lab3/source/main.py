from pathlib import Path

from students.gopienko_ad.lab3.source.dataset import load_dataset, analyze_dataset, train_val_test_split

RESULT_DIR = Path("result")
PLOTS_DIR = RESULT_DIR / "plots"

def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(output_dir=RESULT_DIR / "data")
    analyze_dataset(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = train_val_test_split(
        df,
        target_column="class",
        val_size=0.15,
        test_size=0.15,
        stratify=True,
        random_state=42,
    )