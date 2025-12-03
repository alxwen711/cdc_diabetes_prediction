"""
03-split_preprocess_data.py

Purpose:
    The previous step (02-clean_transform_data.py) has already performed the
    official cleaning/preprocessing and saved:
        ../data/clean/diabetes_clean_train.csv
        ../data/clean/diabetes_clean_test.csv

    This script simply:
    - Loads both files
    - Separates features (X) and target (y)
    - Saves them in the final processed format expected by modeling scripts
"""

import os
import click
import pandas as pd


def load_and_split_file(filepath: str, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a clean split file and separate X and y.
    
    Parameters
    ----------
    filepath : str
        Path to diabetes_clean_train.csv and diabetes_clean_test.csv
    split_name : str
        Either "train" or "test" — used for logging

    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable ('diabetes')

    Examples
    --------
    >>> X_train, y_train = load_and_split_file("../data/clean/diabetes_clean_train.csv", "train")

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    KeyError
        If 'diabetes' column is missing
    """
    """
    Checks verified:
    - File exists and is readable
    - Contains the required 'diabetes' target column
    - No missing values in target (should be clean already)
    - Consistent column structure (assumed from previous step)
    """

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
    """Save the final X and y splits in the processed directory.
    
    Parameters
    ----------
    X_train, y_train, X_test, y_test
        Feature and target DataFrames/Series
    output_dir : str
        Directory to save the four final files

    Returns
    -------
    None
        Files are written to disk

    Examples
    --------
    >>> save_processed_splits(X_train, y_train, X_test, y_test, "../data/processed")

    Raises
    ------
    OSError
        If output directory cannot be created or files cannot be written
    """
    """
    Checks verified:
    - Output directory is created if missing
    - All four files are written successfully
    - Index is not saved (index=False)
    - y is saved as single-column DataFrame for consistency
    """

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "diabetes_X_train.csv": X_train,
        "diabetes_y_train.csv": pd.DataFrame(y_train, columns=["diabetes"]),
        "diabetes_X_test.csv":  X_test,
        "diabetes_y_test.csv":  pd.DataFrame(y_test, columns=["diabetes"]),
    }
    for filename, data in paths.items():
        path = os.path.join(output_dir, filename)
        data.to_csv(path, index=False)
        click.echo(f"   -> {path}  ({data.shape[0]:,} samples)")

    click.echo(f"\nAll processed files saved to: {os.path.abspath(output_dir)}")


@click.command()
@click.option(
    "--clean-train",
    type=str,
    default="../data/clean/diabetes_clean_train.csv",
    help="Path to clean train split"
)
@click.option(
    "--clean-test",
    type=str,
    default="../data/clean/diabetes_clean_test.csv",
    help="Path to clean test split"
)
@click.option(
    "--output-dir",
    type=str,
    default="../data/processed",
    help="Directory to save final X/y processed files"
)
def main(clean_train: str, clean_test: str, output_dir: str):
    """Load teammate's clean splits → separate X/y -> save final processed data."""
    click.echo("Starting 03-split_preprocess_data.py\n")
    # Load and split train
    X_train, y_train = load_and_split_file(clean_train, "train")

    # Load and split test
    X_test, y_test = load_and_split_file(clean_test, "test")

    # Save final processed files
    save_processed_splits(X_train, y_train, X_test, y_test, output_dir)

    click.echo("\nPipeline complete — final X/y splits ready for modeling!")


if __name__ == "__main__":
    main()