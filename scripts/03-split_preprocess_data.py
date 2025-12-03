"""
03-split_preprocess_data.py

Purpose:
    Load the two clean split files produced by 02-clean_transform_data.py:
        data/clean/diabetes_clean_train.csv
        data/clean/diabetes_clean_test.csv

    Separate features (X) and target (y), then save final processed files:
        data/processed/diabetes_X_train.csv
        data/processed/diabetes_y_train.csv
        data/processed/diabetes_X_test.csv
        data/processed/diabetes_y_test.csv

    Designed to be run from the project root (just like 02-clean_transform_data.py)
"""

import os
import click
import pandas as pd 


def load_and_split_file(filepath: str, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a clean split file and separate X and y."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{split_name.capitalize()} clean file not found: {filepath}")

    df = pd.read_csv(filepath)
    click.echo(f"Loaded clean {split_name} split: {df.shape[0]:,} samples × {df.shape[1]} columns")

    if "diabetes" not in df.columns:
        raise KeyError(f"Target column 'diabetes' not found in {filepath}")

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]

    return X, y


def save_processed_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series,
    output_dir: str
) -> None:
    """Save final X/y splits to processed directory."""
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "diabetes_X_train.csv": X_train,
        "diabetes_y_train.csv": pd.DataFrame({"diabetes": y_train}),
        "diabetes_X_test.csv":  X_test,
        "diabetes_y_test.csv":  pd.DataFrame({"diabetes": y_test}),
    }

    click.echo("\nSaving final processed files:")
    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        data.to_csv(path, index=False)
        click.echo(f"   -> {path}  ({data.shape[0]:,} rows)")

    click.echo(f"\nAll done! Files saved to: {output_dir}/")


@click.command()
@click.option(
    "--clean-train",
    type=str,
    default="data/clean/diabetes_clean_train.csv",
    help="Path to clean train split (from 02-clean_transform_data.py)"
)
@click.option(
    "--clean-test",
    type=str,
    default="data/clean/diabetes_clean_test.csv",
    help="Path to clean test split (from 02-clean_transform_data.py)"
)
@click.option(
    "--output-dir",
    type=str,
    default="data/processed",
    help="Directory to save final processed X/y files"
)
def main(clean_train: str, clean_test: str, output_dir: str):
    """Convert clean splits -> final X/y processed files."""
    click.echo("Starting 03-split_preprocess_data.py (@raymondww)\n")

    X_train, y_train = load_and_split_file(clean_train, "train")
    X_test,  y_test  = load_and_split_file(clean_test,  "test")

    save_processed_splits(X_train, y_train, X_test, y_test, output_dir)

    click.echo("\nPipeline complete — ready for modeling!")


if __name__ == "__main__":
    main()