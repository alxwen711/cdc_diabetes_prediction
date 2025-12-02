"""Load raw CDC diabetes data and save to csv.

This sciprt will load the raw dataset from the UCI ML Repository, 
combine the target and features into one dataframe and save as a 
csv. The default folder and filename is data/raw/diabetes_raw.csv
"""

import pandas as pd
import pandera.pandas as pa
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

    
def validate_x_data(a: int, b: int = 0) -> int:
    """short_description
    
    longer_description
    
    Notes
    -----
    
    Parameters
    ----------
    a : int
        description
    b : int, optional (default = 0)
        description
    
    Returns
    -------
    int
        description
    
    Examples
    --------
    >>> snake_case(a, b)
    output
    
    Raises
    --------
    SomeError
        when some error
    
    """
    """
    Checks verified:
    Correct data file format
    Correct column names
    No empty observations (and by extension Missingness not beyond expected threshold)
    Correct data types in each column
    No outlier or anomalous values

    Maximum allowable ranges for the numeric_features were determined from
    the schema description in https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.
    """

    # Check that X is a Pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X object obtained is not a Pandas Dataframe.")

    # Create a list of expected column names
    column_names = ["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
                    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
                    "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]

    # Create a list of column names that are binary features
    binary_features = ["HighBP","HighChol","CholCheck","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
                    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
                    "NoDocbcCost","DiffWalk","Sex"]

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

    schema.validate(X, lazy = True)

#@click.command()
#@click.option("--filepath", type = str, default = "../data/raw", help = "Directory location to write the combined DataFrame in csv format.")
#@click.option("--filename", type = str, default = "diabetes_raw.csv", help = "Name of the file to write the DataFrame to.")
#@click.option("--label", type = str, default = "diabetes", help = "Name of the target value column in the combined DataFrame.")
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

    

def main():
    X,y = obtain_raw_data()
    save_raw_data(X,y)

if __name__ == "__main__":
    main()