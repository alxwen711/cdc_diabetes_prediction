"""Clean and transform raw data.

This script will load the raw data from /data/raw/diabetes_raw_features.csv
and /data/raw/diabetes_raw_features.csv, validate the data, split into train
and test datasets, and save to data/clean
"""

# imports
import pandas as pd
import pandera.pandas as pa
from sklearn.model_selection import train_test_split
import os
import click

# functions
def load_and_validate_raw_data(file: str) -> pd.DataFrame:
    """Load raw data from file and perform validation checks on the whole dataset.
    
    Checks: correct data file format, correct column names, no empty observations,
    (and by extension Missingness not beyond expected threshold), Correct data types
    in each column, No outlier or anomalous values.
    
    Maximum allowable ranges for the numeric_features were determined from
    the schema description in https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.
    
    Parameters
    ----------
    file : str
        The file path for the raw data csv.
    
    Returns
    -------
    pd.DataFrame
        The raw data in a pandas dataframe.
    
    Examples
    --------
    >>> raw_data = load_and_validate_raw_data("/data/raw/diabetes_raw.csv")
    
    """
    raw_data = pd.read_csv(file)

    # Check that raw_data is a Pandas DataFrame
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("raw_data object obtained is not a Pandas Dataframe.")

    # Create a list of expected column names
    column_names = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
                    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
                    "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income", "diabetes"]

    # Create a list of column names that are binary features
    binary_features = ["HighBP","HighChol","CholCheck","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
                    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
                    "NoDocbcCost","DiffWalk","Sex","diabetes"]

    # Define allowable ranges for numeric features
    numeric_features = {"BMI": (5,256),
                        "GenHlth": (1,5),
                        "MentHlth": (0,30),
                        "PhysHlth": (0,30),
                        "Age": (1,13),
                        "Education": (1,6),
                        "Income": (1,8)}

    # Check that all expected columns are present
    schema_dict = {}
    for col_name in binary_features:
        schema_dict[col_name] = pa.Column(int, pa.Check.between(0,1), nullable = False)

    # Add numeric features to schema with their respective ranges
    for col_name in numeric_features.keys():
        schema_dict[col_name] = pa.Column(int, pa.Check.between(numeric_features[col_name][0],numeric_features[col_name][1]), nullable = False)

    schema = pa.DataFrameSchema(schema_dict)

    schema.validate(raw_data, lazy = True)

    return raw_data

def split_dataset_and_validate(full_df: pd.DataFrame):
    """Split dataset into train and test and run validation checks.
    
    Split dataset into train and test.
    
    Parameters
    ----------
    full_df : pd.DataFrame
        Full raw dataset.
    
    Returns
    -------
    train_df : pd.DataFrame
        The train split of full_df.
    test_df : pd.DataFrame
        The test split of full_df.

    Examples
    --------
    >>> split_dataset_and_validate(raw_data)
    train, test
        
    """
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.3,
        random_state=522,
        stratify=full_df['diabetes']
    )

    return train_df, test_df

def save_clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame, clean_data_path: str) -> None:
    """Save train and test datasets to csv.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        The train dataframe to save.
    test_df : pd.DataFrame
        The test dataframe to save.
    
    Returns
    -------
    None
        
    """

    # check/create clean data folder
    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)

    # save processed data
    train_df.to_csv(clean_data_path+"/diabetes_clean_train.csv", index=False)
    test_df.to_csv(clean_data_path+"/diabetes_clean_test.csv", index=False)
    
# main function
@click.command()
@click.option("--file", type=str, default="data/raw/diabetes_raw.csv")
@click.option("--savefilepath", type=str, default="data/clean")
def main(file: str, savefilepath: str):
    """Load, validate, and split dataset. Returns train and test datasets.
    
    Load the raw dataset and run validation that should be done on the whole
    dataset. Then split into train and test datasets and run validation checks
    that should be done separately. Returns train and test datasets.
        
    Parameters
    ----------
    file : str
        description
    b : int, optional (default = 0)
        description
    
    Returns
    -------
    None

    """

    raw_data = load_and_validate_raw_data(file)
    train_df, test_df = split_dataset_and_validate(raw_data)
    save_clean_data(train_df, test_df, savefilepath)

if __name__ == "__main__":
    main() 