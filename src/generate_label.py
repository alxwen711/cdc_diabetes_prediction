import pandas as pd

def generate_label(row: pd.Series, 
                   feature_name: str = "feature_value", 
                   feature_label: str = "f", 
                   value_name: str = "diabetes",
                   value_label: str = "d") -> str:
    """Helper function to create plot labels for a pd.Dataframe.
    Application of this function is only through df.apply(generate_label, axis = 1)
    where df is a pd.Dataframe.
    
    Parameters
    ----------
    row : pd.Series
        A row of a pandas Dataframe.
    feature_name : str (default = "feature_value")
        The name of the feature to use for labelling.
    feature_label : str (default = "f")
        The label for the feature to use.
    value_name : str (default = "diabetes")
        The name of the feature's value column to use for labelling.
    value_label : str (default = "d")
        The label for the feature's value column to use.
    
    Returns
    -------
    str
        str type label used in the binary bar plots.
    
    Examples
    --------
    >>> df = pd.DataFrame({'feature_value': [1,2], 'diabetes': [3,4]})
    >>> df["label"] = df.apply(generate_label, axis = 1)
    >>> print(df)

       feature_value  diabetes         label
    0              1         3  f = 1, d = 3
    1              2         4  f = 2, d = 4
    """
    return f"{feature_label} = {row[feature_name]}, {value_label} = {row[value_name]}"


if __name__ == "__main__":
    df = pd.DataFrame({'feature_value': [1,2], 'diabetes': [3,4]})
    df["label"] = df.apply(generate_label, axis = 1)
    print(df)