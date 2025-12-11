"""Load raw CDC diabetes data and save to csv.

This sciprt will load the raw dataset from the UCI ML Repository, 
combine the target and features into one dataframe and save as a 
csv. The default folder and filename is data/raw/diabetes_raw.csv
"""

import pandas as pd
import os
from ucimlrepo import fetch_ucirepo 
import click
import sys

import warnings
warnings.filterwarnings("ignore",category=pd.errors.SettingWithCopyWarning)

# expand scope for packages
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_raw_data import save_raw_data


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

    # fetch dataset via the UCI ML Repository
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 

    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets
    
    click.echo(f"Raw CDC data obtained, writing to {filepath}/{filename}...")
    save_raw_data(X, y, filepath, filename, label)
    
    click.echo(f"Raw data extraction to {filepath}/{filename} is complete")

if __name__ == "__main__":
    main()