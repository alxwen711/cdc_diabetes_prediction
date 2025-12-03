import click

# import script 03-split_preprocess_data.py to obtain training data
preprocess_name = "03-split_preprocess_data"
preprocess = __import__(preprocess_name)

def eda_head():
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

def eda_tail():
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

def eda_describe():
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
    help = "Run a specific EDA command from the list of EDA tasks done in order: \n ['head', 'tail', 'describe', 'count', 'histogram', 'boxplot', 'correlation']"
)
@click.option(
    "--save",
    "-s",
    type = str,
    default = None,
    help = "Specify a specific directory and filename to save a generated image to if the EDA command is among the following: \n ['count', 'boxplot', 'histogram', 'correlation']"
)
def run_eda_function(command: str, save = None):
    plot = None
    X_train,y_train = preprocess.load_and_split_file("data/clean/diabetes_clean_train.csv","train")
    print(X_train.head())
    match command:
        case "head":
            click.echo("Printing first few rows of training data...")
            #eda_head(train_df)
        case "tail":
            click.echo("Printing last few rows of training data...")
            #eda_tail(train_df)
        case "describe":
            click.echo("Printing description of training data...")

        case "count":
            click.echo("Creating basic bar plot for count of target labels...")
        case "histogram":
            click.echo("Creating histograms of all features...") # Might be adjusted?
        case "boxplot":
            click.echo("Creating boxplots of all non-binary features...")
        case "correlation":
            click.echo("Creating feature-feature correlation plot on training data...")
        case _:
            click.echo("Command is not recognized from the options ['head', 'tail', 'describe', 'count', 'boxplot', 'histogram', 'correlation'], no action performed")
            return
    click.echo(f"Command {command} compeleted as expected.")
    if save != None and command in ["count", "boxplot", "histogram", "correlation"]:
        click.echo(f"Saving plot to {save}...")
        save_figure(plot,save)
        click.echo(f"Plot has been saved to {save}.")

if __name__ == "__main__":
    run_eda_function()



