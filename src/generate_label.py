import pandas as pd

def generate_label(row: pd.Series) -> str:
    """Helper function to create the x-axis labels for eda_binary.
    
    Parameters
    ----------
    row : pd.Series
        A row of a pandas Dataframe.

    Returns
    -------
    str
        str type label used in the binary bar plots.
    """
    return f"f = {row['feature_value']}, d = {row['diabetes']}"


if __name__ == "__main__":
    df = pd.DataFrame({'feature_value': [1,2], 'diabetes': [3,4]})
    df["label"] = df.apply(generate_label, axis = 1)
    print(df)