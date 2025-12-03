import click
import pandas as pd
import altair as alt
import os

# import script 03-split_preprocess_data.py to obtain training data
preprocess_name = "03-split_preprocess_data"
preprocess = __import__(preprocess_name)

def eda_describe(df):
    """Macro function to output the head, tail, and description (df.describe) of a DataFrame 
    representing the training data for EDA purposes
    
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
    print("First few rows of the training data:")
    print(df.head())
    print("\nLast few rows of the training data:")
    print(df.tail())
    print("\nDescription of the training data:")
    print(df.describe())
    

def eda_count(y_train: pd.Series) -> alt.Chart:
    """
    Creates a plot of the frequency of the target labels in the training data.
    
    Parameters
    ----------
    y_train : pd.Series
        pd.Series object containing all the target labels

    Returns
    -------
    alt.Chart
        Frequency bar graph of the labels.
    
    Examples
    --------
    >>> snake_case(a, b)
    output
    """
    diabetes_count = pd.DataFrame(y_train.value_counts()).reset_index()

    chart = alt.Chart(diabetes_count).mark_bar().encode(
        x=alt.X('diabetes:O', title='Has Diabetes'),
        y="count",
        color="diabetes:N"
    ).properties(title='Count of Diabetes vs Non-Diabetes Records in Dataset')

    return chart

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


def save_figure(plot: alt.Chart, filename: str,  filepath: str = "img"):
    """
    Saves a Altair Chart created from a previous EDA function to the given path under a specified name.
    
    Parameters
    ----------
    plot: alt.Chart
        The Altair Chart created from a previous EDA function.
    filename: str
        The filename to save the image to.
    filepath: str, default = "img"
        The directory path to save the chart to. The default option
        will save the plot to the `img` folder relative to the root 
        directory, creating it if it does not currently exist.
    
    Examples
    --------
    >>> plot = alt.Chart(...)
    >>> save_figure(plot,"count.png","img")
    """
    
    if filepath == None: filepath = "img"

    # check if folder exists
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Save Raw Data
    plot.save(os.path.join(filepath,filename))


command_options = ['describe', 'count', 'histogram', 'boxplot', 'correlation', 'saveallcharts']
plot_options = ["count", "boxplot", "histogram", "correlation"]


@click.command()
@click.option(
    "--command",
    "-c", 
    type = str, 
    default = "count", 
    help = f"Run a specific EDA command from the list of EDA tasks done in order: \n {command_options}"
)
@click.option(
    "--path",
    "-p",
    type = str,
    default = None,
    help = f"Specify a specific directory to save a generated image to if the EDA command is among the following: \n {plot_options} \n 'saveallcharts' (special command that does all of the above in one query)"
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
            plot = eda_count(y_train)
        
        case "histogram":
            click.echo("Creating histograms of all features...") # Might be adjusted?
        
        case "boxplot":
            click.echo("Creating boxplots of all non-binary features...")
        
        case "correlation":
            click.echo("Creating feature-feature correlation plot on training data...")
            plot = eda_correlation(X_train)
        
        case "saveallcharts":
            click.echo(f"Creating and saving ALL EDA charts to directory {path}...")
            for po in plot_options:
                run_eda_function(po,path)
        
        case _:
            click.echo("Command is not recognized from the options ['describe', 'count', 'boxplot', 'histogram', 'correlation', 'saveallcharts'], no action performed")
            return
    
    click.echo(f"Command {command} compeleted as expected.")
    if path != None and command in plot_options:
        click.echo(f"Saving plot to {path}...")
        save_figure(plot,command+".png",path)
        click.echo(f"Plot has been saved to {path}.")


if __name__ == "__main__":
    run_eda_function()
    


