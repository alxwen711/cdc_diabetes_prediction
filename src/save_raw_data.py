import pandas as pd
import os
from ucimlrepo import fetch_ucirepo 
import click

def save_raw_data(X: pd.DataFrame, y: pd.DataFrame, filepath: str = "data/raw", filename: str = "diabetes_raw.csv", label: str = "diabetes"):
    """
    Writes the raw analysis dataframes to a single file. If the
    file path does not exist they will be created by this function.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame corresponding to the features of the data set.
    y : pd.DataFrame
        DataFrame corresponding to the labels of the data set. y must have exactly one column.
    filepath: str, default = "data/raw"
        Directory location to write the combined DataFrame in csv format.
    filename: str, default = "diabetes_raw.csv"
        Name of the file to write the DataFrame to.
    label: str, default = "diabetes"
        Name of the target value column in the combined DataFrame.
    
    Examples
    --------
    >>> X,y = obtain_raw_data()
    >>> save_raw_data(X, y, "data/raw", "diabetes_raw.csv", "diabetes")
    
    Raises
    --------
    TypeError
        Either X or y is not a valid DataFrame.
    ValueError
        Either y is not a one column DataFrame or the attempt to combine the DataFrames failed, the full error is additionally outputted.
    """

    # Try creating the combined DataFrame
    if not isinstance(X, pd.DataFrame): raise TypeError("X object obtained is not a Pandas Dataframe.")
    if not isinstance(y, pd.DataFrame): raise TypeError("y object obtained is not a Pandas Dataframe.")

    # dimension error checking
    if y.shape[1] != 1: raise ValueError(f"y Dataframe's shape {y.shape} is incompatible for save_raw_data, must be exactly one column.")
    if X.shape[0] != y.shape[0]: raise IndexError(f"X contains {X.shape[0]} but y contains {y.shape[0]}, incompatible shapes")

    data = X

    try:
        data[label] = y
    except Exception as e:
        raise ValueError(f"Attempt to combine the X and y Dataframes failed due to the following: {e}")

    # check if raw folder exists
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Save Raw Data
    data.to_csv(os.path.join(filepath,filename), index = False)