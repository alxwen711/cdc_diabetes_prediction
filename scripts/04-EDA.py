import click
import pandas as pd
import altair as alt
import os

# import script 03-split_preprocess_data.py to obtain training data
preprocess_name = "03-split_preprocess_data"
preprocess = __import__(preprocess_name)

def eda_describe(df: pd.DataFrame) -> list:
    """Macro function to output the head, tail, and description (df.describe) of a DataFrame 
    representing the training data for EDA purposes
    
    Parameters
    ----------
    df : pd.DataFrame
        Assummed training data
    
    Returns
    -------
    List[pd.DataFrame]
    List of 3 dataframes consisting of the head, tail, and description of the DataFrame.
    
    
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
    return [df.head(), df.tail(), df.describe().round(4)] # avoid long decimals from messing up pdf format
    

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
    """
    diabetes_count = pd.DataFrame(y_train.value_counts()).reset_index()

    chart = alt.Chart(diabetes_count).mark_bar().encode(
        x=alt.X('diabetes:O', title='Has Diabetes'),
        y="count",
        color="diabetes:N"
    ).properties(title='Count of Diabetes vs Non-Diabetes Records in Dataset')

    return chart

def eda_histogram(X_train: pd.DataFrame) -> alt.Chart:
    """
    Creates a set of histograms of all non-binary features in the training data set.

    Parameters
    ----------
    X_train : pd.DataFrame
        pd.DataFrame object containing the training data.

    Returns
    -------
    alt.Chart
        Faceted chart of histograms for each non-binary feature.
    """
    features = X_train.columns.to_list()
    df_long = pd.melt(X_train, id_vars=["diabetes"], value_vars=features[:-1], var_name="feature", value_name="feature_value")
    non_binary_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    df_sample_nonbinary = df_long[df_long["feature"].isin(non_binary_features)]

    chart = alt.Chart(df_sample_nonbinary).mark_bar().encode(
        x=alt.X("feature_value:O"), # Chose to use ordinal instead of quantitative because this works better for most features
        y=alt.Y("count()", title="Count").stack(False),
        color=alt.Color("diabetes:N"),
    ).properties(
        width=500,
        height=150,
    ).facet(
        "feature:N",
        columns=1,
    ).resolve_scale(
        x="independent",
        y="independent",   
    ).properties(
        title='Histograms of Non-Binary Features',
    )

    return chart

def generate_label(row) -> str:
    """Helper function to create the x-axis labels for eda_binary.
    
    Parameters
    ----------
    row : pd.Row
        A row of a pandas Dataframe.

    Returns
    -------
    str
        str type label used in the binary bar plots.
    """
    return f"f = {row['feature_value']}, d = {row['diabetes']}"

def eda_binary(X_train: pd.DataFrame) -> alt.Chart:
    """
    Creates a set of bar plots of all binary features in the training data set.

    Parameters
    ----------
    X_train : pd.DataFrame
        pd.DataFrame object containing the training data.

    Returns
    -------
    alt.Chart
        Faceted chart of bar plots for each binary feature.
    """
    features = X_train.columns.to_list()
    df_long = pd.melt(X_train, id_vars=["diabetes"], value_vars=features[:-1], var_name="feature", value_name="feature_value")
    binary_features = ["HighBP","HighChol","CholCheck","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
                "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
                "NoDocbcCost","DiffWalk","Sex"]
    df_sample_binary = df_long[df_long["feature"].isin(binary_features)]
    df_sample_binary["label"] = df_sample_binary.apply(generate_label, axis = 1)
    chart = alt.Chart(df_sample_binary).mark_bar().encode(
        x=alt.Y("count()", title="Count").stack(False),
        y=alt.X("label"), 
        color=alt.Color("diabetes:N"),
    ).properties(
        width=150,
        height=150,
    ).facet(
        "feature:N",
        columns=3,
    ).resolve_scale(
        x="independent",
        y="independent",   
    ).properties(
        title='Bar Plots of Binary Features',
    )

    return chart


def eda_boxplot(X_train: pd.DataFrame) -> alt.Chart:
    """
    Creates a set of boxplots of all non-binary features in the training data set.
    A random sample of 1000 observations is used for generating these plots due to
    program runtime constraints and to not oversatuarate the box plots.

    Parameters
    ----------
    X_train : pd.DataFrame
        pd.DataFrame object containing the training data.

    Returns
    -------
    alt.Chart
        Faceted chart of boxplots for each non-binary feature.
    
    """
    features = X_train.columns.to_list()
    df_long = pd.melt(X_train, id_vars=["diabetes"], value_vars=features[:-1], var_name="feature", value_name="feature_value")
    non_binary_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    df_sample_nonbinary = df_long[df_long["feature"].isin(non_binary_features)]

    chart = alt.Chart(df_sample_nonbinary,).mark_boxplot().encode(
        x='diabetes:N',
        y='feature_value:Q',
        color='diabetes:N'
    ).properties(
        width = 65,
        height = 300
    ).facet(
        column='feature:N'
    ).properties(
        title ='Boxplots for Non-Binary Features (Sample size n = 1000)'
    ).resolve_scale(
        y="independent"
    )
    return chart


def eda_correlation(X_train):
    """
    Creates a feature-feature correlation heatmap of the training data set.

    Parameters
    ----------
    X_train : pd.DataFrame
        pd.DataFrame object containing the training data.

    Returns
    -------
    alt.Chart
        Feature-feature correlation heatmap of the training data.
    """
    correlation_matrix = X_train.corr()
    correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

    chart = alt.Chart(correlation_long).mark_rect().encode(
        x='Feature 1:O',
        y='Feature 2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(domain = [-1,1], scheme='spectral')),
        tooltip=['Feature 1', 'Feature 2', 'Correlation']
    ).properties(
        width=600,
        height=600,
        title="Correlation Heatmap"
    )
    return chart



def save_figure(plot: alt.Chart, filename: str,  filepath: str = "results/figures"):
    """
    Saves a Altair Chart created from a previous EDA function to the given path under a specified name.
    
    Parameters
    ----------
    plot: alt.Chart
        The Altair Chart created from a previous EDA function.
    filename: str
        The filename to save the image to.
    filepath: str, default = "results/figures"
        The directory path to save the chart to. The default option
        will save the plot to the `results/figures` folder relative to the root 
        directory, creating it if it does not currently exist.
    
    Examples
    --------
    >>> plot = alt.Chart(...)
    >>> save_figure(plot,"count.png","img")
    """

    if filepath == None: filepath = "results/figures"

    # check if folder exists
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Save Plot
    plot.save(os.path.join(filepath,filename))

def save_dataframe(df: pd.DataFrame, filename: str, filepath: str = "results/tables"):
    """
    Saves a Pandas DataFrame created from a previous EDA function to the given path under a specified name.
    
    Parameters
    ----------
    df: pd.DataFrame
        The Pandas Dataframe created from a previous EDA function.
    filename: str
        The filename to save the DataFrame to.
    filepath: str, default = "results/tables"
        The directory path to save the chart to. The default option
        will save the plot to the `results/tables` folder relative to the root 
        directory, creating it if it does not currently exist.
    
    Examples
    --------
    >>> df = alt.Chart(...)
    >>> save_figure(df,"EDA.csv","dataframes")
    """
    if filepath == None: filepath = "results/tables"

    # check if folder exists
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Save Raw Data
    if filename == "EDA_describe.csv": df.to_csv(os.path.join(filepath,filename),index = True) # index is needed for description
    else: df.to_csv(os.path.join(filepath,filename),index = False) # index should be removed for examples


command_options = ['describe', 'count', 'histogram', 'binary', 'boxplot', 'correlation', 'saveallcharts']

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
    help = f"Specify a specific directory to save the results to. If no option is specified the results will not be saved."
)
def run_eda_function(command: str, path = None):
    plot = None
    X_train,y_train = preprocess.load_and_split_file("data/clean/diabetes_clean_train.csv","train")
    train_df = X_train
    train_df["diabetes"] = y_train
    
    match command:
        
        case "describe":
            click.echo("Printing description of training data...")
            plot = eda_describe(train_df)
        
        case "count":
            click.echo("Creating basic bar plot for count of target labels...")
            plot = eda_count(y_train)
        
        case "histogram":
            click.echo("Creating histograms of all features...")
            plot = eda_histogram(X_train)

        case "binary":
            click.echo("Creating bar plots of all binary features...")
            plot = eda_binary(X_train)
        
        case "boxplot":
            click.echo("Creating boxplots of all non-binary features from a random sample of 10000 observations...")
            df_sample = X_train.sample(n=1000, random_state=522)
            plot = eda_boxplot(df_sample)

        case "correlation":
            click.echo("Creating feature-feature correlation plot on training data...")
            plot = eda_correlation(X_train)
        
        case "saveallcharts":
            # Macro command to run ALL plots (to be more convinient for the script, figure out recursion for this step if possible)
            click.echo(f"Creating and saving ALL EDA charts to directory {path}...")
            
            # count
            click.echo("Creating basic bar plot for count of target labels...")
            plot = eda_count(y_train)
            save_figure(plot,"EDA_count.png",path)

            # histogram
            click.echo("Creating histograms of all features...")
            plot = eda_histogram(X_train)
            save_figure(plot,"EDA_histogram.png",path)

            # boxplot
            click.echo("Creating boxplots of all non-binary features from a random sample of 10000 observations...")
            df_sample = X_train.sample(n=1000, random_state=522)
            plot = eda_boxplot(df_sample)
            save_figure(plot,"EDA_boxplot.png",path)

            # correlation
            click.echo("Creating feature-feature correlation plot on training data...")
            plot = eda_correlation(X_train)
            save_figure(plot,"EDA_correlation.png",path)

            # binary
            click.echo("Creating bar plots of all binary features...")
            plot = eda_binary(X_train)
            save_figure(plot,"EDA_binary.png",path)

        case _:
            click.echo("Command is not recognized from the options ['describe', 'count', 'boxplot', 'histogram', 'correlation', 'saveallcharts'], no action performed")
            return
    
    click.echo(f"Command {command} compeleted as expected.")

    if path != None and command != "saveallcharts": # single option

        if command == "describe": # use csv format
            click.echo(f"Saving 3 EDA Dataframes to {path}...")
            describe_name = ["head","tail","describe"]
            for i in range(3):
                click.echo(f"Saving the {describe_name[i]} DataFrame to {path}...")
                save_dataframe(plot[i],"EDA_"+describe_name[i]+".csv",path)
            click.echo(f"All description DataFrames for the training data have been saved to {path}.")
        
        else: # use png format
            click.echo(f"Saving plot to {path}...")
            save_figure(plot,"EDA_"+command+".png",path)
            click.echo(f"Plot has been saved to {path}.")


if __name__ == "__main__":
    run_eda_function()
    


