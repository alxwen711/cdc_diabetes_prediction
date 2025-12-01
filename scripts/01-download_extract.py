import pandas as pd
import pandera.pandas as pa
import os
from ucimlrepo import fetch_ucirepo 



def raw_data():
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


def save_raw_data(a: int, b: int = 0) -> int:
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
    # check if raw folder exists
    raw_data_path = "../data/raw"

    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    ## Save Raw Data
    X.to_csv("../data/raw/diabetes_raw_features.csv")
    y.to_csv("../data/raw/diabetes_raw_targets.csv")


def main():
    return

if __name__ == "__main__":
    main()