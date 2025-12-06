"""Clean and transform raw data.

This script will load the raw data from /data/raw/diabetes_raw_features.csv
and /data/raw/diabetes_raw_features.csv, validate the data, split into train
and test datasets, and save to data/clean
"""

# imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pandera.pandas as pa
from sklearn.model_selection import train_test_split
import os
import click
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset


# functions
def iqr_outliers(series: pd.Series) -> bool:
    """Return True if all values are within 1.5 * IQR (no extreme outliers)
    
    Used for data validation

    Parameters
    ----------
    series : pd.Series
        The series to do the data validation on
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.between(lower_bound, upper_bound).all()

def load_and_validate_raw_data(file: str) -> pd.DataFrame:
    """Load raw data from file and perform validation checks on the whole dataset.
    
    Perform these data validation checks:
    - Correct data file format
    - Correct column names
    - No empty observations
    - Correct data types
    - No outliers or anamalour values
    - Checking if duplicate rows exist in the dataset, preventing redundant data points.
      (We are not dropping duplicates automatically to avoid unintentional data loss.)
      (For the purpose of this dataset, because we already finished the analysis, we will accept the duplicate if any exist.)
    - Checking for outlier in ("BMI", "MentHlth", "PhysHlth") columns to see if they have extreme outliers based on the IQR rule,
      which helps maintain data integrity and model robustness.
      (For the purpose of this dataset, we will accept the outlier value because it is a valid measurement.)
    - Categorical validation is not applicable for this dataset
    
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

    # Define the schema with meaningful checks
    diabetes_schema = pa.DataFrameSchema(
        columns={
            "BMI": pa.Column(float, nullable=False),
            "MentHlth": pa.Column(float, nullable=False),
            "PhysHlth": pa.Column(float, nullable=False),
        },
        checks=[
            # 1. No duplicate rows
            pa.Check(
                lambda df: ~df.duplicated().any(),
                error="DUPLICATE_ROWS: Found duplicate observations. Use df.drop_duplicates() to remove them."
            ),

            # 2. No extreme outliers in continuous variables using IQR rule
            pa.Check(
                lambda df: iqr_outliers(df["BMI"]),
                error="OUTLIERS_IN_BMI: Extreme outliers detected in BMI (beyond 1.5 × IQR). Consider winsorizing or removing."
            ),
            pa.Check(
                lambda df: iqr_outliers(df["MentHlth"]),
                error="OUTLIERS_IN_MENTHLTH: Extreme values in MentHlth (beyond 1.5 × IQR)."
            ),
            pa.Check(
                lambda df: iqr_outliers(df["PhysHlth"]),
                error="OUTLIERS_IN_PHYSHLTH: Extreme values in PhysHlth (beyond 1.5 × IQR)."
            )
        ]
    )

    # Validate with lazy=True to see all errors at once
    # try:
    #     diabetes_schema.validate(raw_data, lazy=True)
    #     print("All checks passed! Dataset is clean and ready for modeling.")
    # except pa.errors.SchemaErrors as e:
    #     print("Validation failed! See errors below:")
    #     print(e.failure_cases)  # Shows detailed failure report

    return raw_data

def split_dataset_and_validate(full_df: pd.DataFrame):
    """Split dataset into train and test and run validation checks.
    
    Split dataset into train and test.

    Data validation:
    - Target/response variable follows expected distribution
      (In the diabetes dataset, we expect the prevalence of diabetes to be around 13-15%.)

    Validate training data for anomalous correlations:
    - Feature-label correlations (target vs features)
    - Feature-feature correlations (between features)
    
    Thresholds set based on domain knowledge.
    
    We perform these checks on training data only because including test data here could lead to data leakage 
    and invalidate the evaluation of model generalization.
    
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

    #Target variable follows expected distribution
    #In this dataset: ~13–15% diabetes is normal, >30% or <5% is suspicious
    
    distribution_schema = pa.DataFrameSchema(
        checks=[pa.Check(
            lambda df: df.mean().between(0.05, 0.30),
            error="ANOMALOUS_TARGET_DISTRIBUTION: Diabetes prevalence should be 5–30% (actual: {:.1%})"
            )
        ]
    )
    
    distribution_schema.validate(train_df[['diabetes']], lazy=True)
    
    #Combine features and target into a dataset for validation
    diabetes_train_ds = Dataset(
        df=train_df,   
        label="diabetes",                          
        cat_features=[]
    )
    
    #Check that feature-label predictive power score (PPS) is below 0.9
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(diabetes_train_ds)
    
    #Check that no feature pairs have correlation above 0.92
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.92, n_pairs=0   
    )
    check_feat_feat_corr_result = check_feat_feat_corr.run(diabetes_train_ds)
    
    #Raise errors if any checks fail
    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")
    
    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")
  
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