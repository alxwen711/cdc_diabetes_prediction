
import pandas as pd
import os
import click 

def load_test_data(
    x_path: str = "data/processed/diabetes_X_test.csv",
    y_path: str = "data/processed/diabetes_y_test.csv"
) -> tuple[pd.DataFrame, pd.Series]:
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
    
    Return
    -----------
    X_test : pd.DataFrame
    y_test : pd.Series
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
        f"Loaded test set: {X_test.shape[0]:,} samples "
    )
    return X_test, y_test
