import click
import pandas as pd
# import script 03-split_preprocess_data.py to obtain training data
preprocess_name = "03-split_preprocess_data"
preprocess = __import__(preprocess_name)

def eda_describe(df):
    """Macro function to output the head, tail, and description (df.describe) of a DataFrame for EDA purposes
    
    Parameters
    ----------
    df : pd.DataFrame
        Assummed training data
    
    Examples
    --------
    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> eda_describe(df)
    
       col1  col2
    0     1     3
    1     2     4


       col1  col2
    0     1     3
    1     2     4


               col1      col2
    count  2.000000  2.000000
    mean   1.500000  3.500000
    std    0.707107  0.707107
    min    1.000000  3.000000
    25%    1.250000  3.250000
    50%    1.500000  3.500000
    75%    1.750000  3.750000
    max    2.000000  4.000000
    """
    print(df.head())
    print("\n")
    print(df.tail())
    print("\n")
    print(df.describe())
    

def eda_count():
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

def eda_histogram():
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

def eda_boxplot():
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

def eda_correlation():
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

def save_figure(plot, path: str):
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




@click.command()
@click.option(
    "--command",
    "-c", 
    type = str, 
    default = "count", 
    help = "Run a specific EDA command from the list of EDA tasks done in order: \n ['describe', 'count', 'histogram', 'boxplot', 'correlation', 'saveallcharts']"
)
@click.option(
    "--path",
    "-p",
    type = str,
    default = None,
    help = "Specify a specific directory to save a generated image to if the EDA command is among the following: \n ['count', 'boxplot', 'histogram', 'correlation', 'saveallcharts']"
)
def run_eda_function(command: str, path = None):
    plot = None
    X_train,y_train = preprocess.load_and_split_file("data/clean/diabetes_clean_train.csv","train")
    train_df = X_train
    train_df["diabetes"] = y_train
    
    match command:
        case "describe":
            click.echo("Printing description of training data...")
            eda_describe(train_df)
        case "count":
            click.echo("Creating basic bar plot for count of target labels...")
        case "histogram":
            click.echo("Creating histograms of all features...") # Might be adjusted?
        case "boxplot":
            click.echo("Creating boxplots of all non-binary features...")
        case "correlation":
            click.echo("Creating feature-feature correlation plot on training data...")
        case "saveallcharts":
            click.echo(f"Creating and saving ALL EDA charts to directory {path}...")
        case _:
            click.echo("Command is not recognized from the options ['describe', 'count', 'boxplot', 'histogram', 'correlation', 'saveallcharts'], no action performed")
            return
    click.echo(f"Command {command} compeleted as expected.")
    if path != None and command in ["count", "boxplot", "histogram", "correlation"]:
        click.echo(f"Saving plot to {path}...")
        save_figure(plot,path,command)
        click.echo(f"Plot has been saved to {path}.")

"""
TODO:

Currently run EDA function is pulling up the X_train and y_train for each individual function, either:
- rework the interface to load data first and then run multiple commands at a time (see next points for changes)
- add additional command to make all of the plots at once
- combine head/tail/describe blocks into a single function (they appear sequentially in the ipynb already)

- assuming above template is accurate, then use ipynb code to fill in the gaps
"""


if __name__ == "__main__":
    run_eda_function()
    


