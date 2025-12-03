import pandas as pd
import os
from ucimlrepo import fetch_ucirepo 
import click


def obtain_raw_data():
    """
    Obtains the raw CDC diabetes data for the analysis via the UCI ML Repository.
    
    Returns
    -------
    pd.Dataframe, pd.Dataframe
        Two data frames, the first corresponding to the [X] features of the data set,
        and the second corresponding to the [y] target labels of the data set.
    
    Examples
    --------
    >>> obtain_raw_data()
    pd.Dataframe, pd.Dataframe
    """
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 

    return X,y

    
def save_raw_data(X: pd.DataFrame, y: pd.DataFrame, filepath: str = "data/raw", filename: str = "diabetes_raw.csv", label: str = "diabetes"):
    """
    Writes the raw analysis dataframes to a single file. If the
    file path does not exist they will be created by this function.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame corresponding to the features of the data set.
    y : pd.DataFrame
        DataFrame corresponding to the labels of the data set.
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
        Attempt to combine the DataFrames failed, the full error is additionally outputted.
    """

    # Try creating the combined DataFrame
    if not isinstance(X, pd.DataFrame): raise TypeError("X object obtained is not a Pandas Dataframe.")
    if not isinstance(y, pd.DataFrame): raise TypeError("y object obtained is not a Pandas Dataframe.")

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

@click.command()
@click.option(
    "--filepath", 
    type = str, 
    default = "data/raw", 
    help = "Directory location to write the combined DataFrame in csv format."
)
@click.option(
    "--filename", 
    type = str, 
    default = "diabetes_raw.csv", 
    help = "Name of the csv file to write the DataFrame to."
)
@click.option(
    "--label", 
    type = str, 
    default = "diabetes", 
    help = "Name of the target value column in the combined DataFrame."
)
def main(filepath: str = "data/raw", filename: str = "diabetes_raw.csv", label: str = "diabetes"):
    """Obtain raw CDC diabetes data and save to a single raw csv file. Default location is data/raw/diabetes_raw.csv."""
    
    click.echo("Obtaining raw CDC data...")
    X,y = obtain_raw_data()
    
    click.echo(f"Raw CDC data obtained, writing to {filepath}/{filename}...")
    save_raw_data(X, y, filepath, filename, label)
    
    click.echo(f"Raw data extraction to {filepath}/{filename} is complete")

if __name__ == "__main__":
    main()