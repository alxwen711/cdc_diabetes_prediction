"""Clean and transform raw data.

This script will load the raw data from /data/raw/diabetes_raw_features.csv
and /data/raw/diabetes_raw_features.csv
"""

# import libraries/packages
import pandas as pd
import pandera as pa
import click

# functions
def load_and_validate_raw_data(file: str):
    """Load raw data from file and perform validation checks on the whole dataset.
    
    Checks: correct data file format, correct column names, no empty observations,
    (and by extension Missingness not beyond expected threshold), Correct data types
    in each column, No outlier or anomalous values.
    
    Maximum allowable ranges for the numeric_features were determined from
    the schema description in https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.
    
    Parameters
    ----------
    file : str
        The file path for the raw data csv
    
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

def split_dataset_and_validate():
    pass

def save_clean_data():
    pass



# main function
@click.command()
@click.option("--file", default="/data/raw/diabetes_raw.csv")
def main():
    # code for "guts" of script goes here

# call main function
if __name__ == "__main__":
    main() # pass any command line args to main here