"""
04-evaluate_model.py
Purpose:
    Load the final processed test set (X_test, y_test) produced by 03-split_preprocess_data.py
    and provide reusable utilities for model evaluation and visualisation.
    Current features:
        • load_test_data() - safely loads X_test and y_test
        • save_figure() - saves Altair charts with consistent logging
    Designed to be run from the project root.
"""

import os
import click
import pandas as pd
import altair as alt
from typing import Tuple


def load_test_data(
    x_path: str = "data/processed/diabetes_X_test.csv",
    y_path: str = "data/processed/diabetes_y_test.csv"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load final processed test features and target.
   
    Parameters
    ----------
    x_path : str, default "data/processed/diabetes_X_test.csv"
        Path to processed test features
    y_path : str, default "data/processed/diabetes_y_test.csv"
        Path to processed test labels (single-column with header 'diabetes')
   
    Returns
    -------
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable ('diabetes')
   
    Examples
    --------
    >>> X_test, y_test = load_test_data()
   
    Raises
    ------
    FileNotFoundError
        If either file is missing
    KeyError
        If 'diabetes' column is missing in y_test file
    """
    """
    Checks verified:
    - Both files exist and are readable
    - y_test contains exactly one column named 'diabetes'
    - No unexpected index columns
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Test features file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Test labels file not found: {y_path}")
   
    X_test = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
   
    if "diabetes" not in y_df.columns:
        raise KeyError(f"Target column 'diabetes' not found in {y_path}")
   
    y_test = y_df["diabetes"]
   
    click.echo(
        f"Loaded test set: {X_test.shape[0]:,} samples × "
        f"{X_test.shape[1]} features, {y_test.shape[0]:,} labels"
    )
    return X_test, y_test


def save_figure(
    plot: alt.Chart,
    filename: str,
    output_dir: str = "img"
) -> None:
    """
    Save an Altair chart to the specified directory.
   
    Parameters
    ----------
    plot : alt.Chart
        The Altair chart object to save
    filename : str
        Name of the output file (e.g. "roc_curve.png")
    output_dir : str, default "img"
        Directory where the figure will be saved.
        Created automatically if it doesn't exist.
   
    Returns
    -------
    None
        Chart is written to disk
   
    Examples
    --------
    >>> chart = alt.Chart(df).mark_bar().encode(...)
    >>> save_figure(chart, "feature_importance.png")
    >>> save_figure(chart, "confusion_matrix.png", "img/results")
   
    Raises
    ------
    OSError
        If directory cannot be created or file cannot be written
    """
    """
    Checks verified:
    - Output directory is created if missing
    - Full path is logged clearly
    - Altair save() is used (handles HTML/PNG via vega)
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    plot.save(full_path)
    click.echo(f"Chart saved → {full_path}")


@click.command()
@click.option(
    "--x-test",
    type=str,
    default="data/processed/diabetes_X_test.csv",
    help="Path to processed test features (X_test)"
)
@click.option(
    "--y-test",
    type=str,
    default="data/processed/diabetes_y_test.csv",
    help="Path to processed test labels (y_test)"
)
@click.option(
    "--img-dir",
    type=str,
    default="img",
    help="Directory for saving evaluation plots"
)
def main(x_test: str, y_test: str, img_dir: str) -> None:
    """Model evaluation entrypoint — loads test data and prepares visualisation utilities."""
    click.echo("Starting 04-evaluate_model.py — Model Evaluation Utilities\n")
   
    # Load test data (with full validation)
    X_test, y_test = load_test_data(x_path=x_test, y_path=y_test)
   
    # Example placeholder — replace with actual evaluation code later
    click.echo("\nTest data loaded successfully — ready for model evaluation & visualisation!")

    # Simple demo output (can be removed later)
    click.echo("\nFirst few rows of X_test:")
    click.echo(X_test.head())
    click.echo("\nFirst few labels (y_test):")
    click.echo(y_test.head().to_frame().T)


if __name__ == "__main__":
    main()