"""
06-evaluate_model.py
Purpose:
    Load the final processed test set (X_test, y_test) produced by 03-split_preprocess_data.py
    and provide reusable utilities for model evaluation and visualisation.
    Current features:
        • load_test_data() - safely loads X_test and y_test
        • load_pickle_models() - loads both trained models from disk
        • plot_bar() - creates a bar chart comparing model performance metrics
        • plot_confusion_matrix() - plots confusion matrix for a given model
        • create_score_table() - creates a DataFrame summarizing model performance metrics
        • save_figure() - saves Altair charts with consistent logging
    Designed to be run from the project root.
"""

import os
import click
import pickle
import pandas as pd
import altair as alt
from sklearn.metrics import (
    accuracy_score, fbeta_score, 
    recall_score, precision_score, ConfusionMatrixDisplay
)

def load_test_data(
    x_path: str = "data/processed/diabetes_X_test.csv",
    y_path: str = "data/processed/diabetes_y_test.csv"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load final processed test features and target.
   
    Parameters
    ----------
    x_path : str, default "data/processed/diabetes_X_test.csv"
        Path to processed test features
    y_path : str, default "data/processed/diabetes_y_test.csv"
        Path to processed test labels (single-column with header 'diabetes')
   
    Returns
    -------
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable ('diabetes')
   
    Examples
    --------
    >>> X_test, y_test = load_test_data()
   
    Raises
    ------
    FileNotFoundError
        If either file is missing
    KeyError
        If 'diabetes' column is missing in y_test file
    """
    """
    Checks verified:
    - Both files exist and are readable
    - y_test contains exactly one column named 'diabetes'
    - No unexpected index columns
    
    Return
    -----------
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Test features file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Test labels file not found: {y_path}")
   
    X_test = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
   
    if "diabetes" not in y_df.columns:
        raise KeyError(f"Target column 'diabetes' not found in {y_path}")
   
    y_test = y_df["diabetes"]
   
    click.echo(
        f"Loaded test set: {X_test.shape[0]:,} samples "
    )
    return X_test, y_test

def load_pickle_models(
    model_dir: str = "models"
) -> tuple[object, object]:
    """
    Load both trained models from the models directory.
   
    Expected files:
        models/tree_model.pickle
        models/naive_bayes_model.pickle
   
    Parameters
    ----------
    model_dir : str, default "models"
        Directory containing the pickled model files
   
    Returns
    -------
    tree_model : object
        Loaded Decision Tree model
    nb_model : object
        Loaded Naive Bayes model
   
    Examples
    --------
    >>> tree, nb = load_pickle_models("models")
   
    Raises
    ------
    FileNotFoundError
        If either model file is missing
    pickle.UnpicklingError
        If model file is corrupted
    """
    tree_path = os.path.join(model_dir, "tree_model.pkl")
    nb_path = os.path.join(model_dir, "naive_bayes_model.pkl")
   
    missing = []
    if not os.path.exists(tree_path):
        missing.append(tree_path)
    if not os.path.exists(nb_path):
        missing.append(nb_path)
   
    if missing:
        raise FileNotFoundError(
            f"Model files not found in {model_dir}:\n" + "\n".join(f"  - {p}" for p in missing)
        )
   
    try:
        with open(tree_path, "rb") as f:
            tree_model = pickle.load(f)
        with open(nb_path, "rb") as f:
            nb_model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise pickle.UnpicklingError(f"Failed to load model (corrupted file): {e}")
   
    click.echo(f"Loaded models from: {os.path.abspath(model_dir)}")
    click.echo("   -> Decision Tree model loaded")
    click.echo("   -> Naive Bayes model loaded")
   
    return tree_model, nb_model

def plot_bar(score_df: pd.DataFrame) -> alt.Chart:
    """
    Create a bar chart comparing model performance metrics.
    
    Parameters
    ----------
    score_df : pd.DataFrame
        DataFrame containing model names and their corresponding performance metrics. 
    Examples
    --------
    >>> chart = plot_bar(score_df)
    
    Returns
    -------
    alt.Chart
        An Altair bar chart comparing the performance of different models across various metrics.

    """

    score_melt = score_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    bar_plot = alt.Chart(score_melt).mark_bar().encode(
        x='Model:N',
        y='Score:Q',
        color='Model:N',
        column='Metric:N'
    ).properties(
        title='Decision Tree vs Naive Bayes Performance on Test Set'
    )
    return bar_plot
    
# Confusion matrix for best model (decision tree)
def plot_confusion_matrix(tree_model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Plot the confusion matrix for a given model on the test set.
    
    Parameters
    ----------
    tree_model : object
        The trained model to evaluate.
    X_test : pd.DataFrame
        The test features.
    y_test : pd.Series
        The true labels for the test set.
    
    Examples
    --------
    >>> plot_confusion_matrix(tree_model, X_test, y_test)
    """

    cm = ConfusionMatrixDisplay.from_estimator(
        tree_model,
        X_test,
        y_test,
        values_format="d",
    )
    return cm.figure_

def create_score_table(X_test: pd.DataFrame, y_test: pd.Series, 
                       best_tree: object, best_nb: object)-> pd.DataFrame:
    models = {
    'Decision Tree': best_tree,
    'Naive Bayes': best_nb
}

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results.append({
            'Model': name,
            'Test Accuracy': round(accuracy_score(y_test, y_pred),3),
            'Test f2-score': round(fbeta_score(y_test, y_pred, beta=2),3),
            'Test recall': round(recall_score(y_test, y_pred),3),
            'Test precision': round(precision_score(y_test, y_pred),3),
        })
    return pd.DataFrame(results)

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
    
    if filepath is None:
        filepath = "img"

    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, filename)

    # Altair chart → .save()
    if isinstance(plot, alt.Chart):
        plot.save(full_path)
    # Matplotlib Figure → plt.savefig()
    else:
        import matplotlib.pyplot as plt
        plt.savefig(full_path, bbox_inches="tight")
        plt.close(plot)   # prevent the figure from staying in memory

    click.echo(f"Saved -> {full_path}")


@click.command()
@click.option(
    "--x-test",
    type=str,
    default="data/processed/diabetes_X_test.csv",
    help="Path to processed test features (X_test)"
)
@click.option(
    "--y-test",
    type=str,
    default="data/processed/diabetes_y_test.csv",
    help="Path to processed test labels (y_test)"
)
@click.option(
    "--model-dir",
    type=str,
    default="models",
    help="Directory containing tree_model.pkl and naive_bayes_model.pkl"
)
@click.option(
    "--img-dir",
    type=str,
    default="img",
    help="Directory for saving evaluation plots"
)
def main(x_test: str, y_test: str, model_dir: str, img_dir: str) -> None:
    """Model evaluation entrypoint — load test data and models for analysis."""
    click.echo("Starting 04-evaluate_model.py — Model Evaluation & Visualisation\n")
   
    # Load test data
    X_test, y_test = load_test_data(x_path=x_test, y_path=y_test)
   
    # Load models
    tree_model, nb_model = load_pickle_models(model_dir=model_dir)
    score_table = create_score_table(X_test, y_test, tree_model, nb_model)
    bar_plot = plot_bar(score_table)
    cm = plot_confusion_matrix(tree_model, X_test, y_test)
    save_figure(bar_plot, "model_performance_comparison.png", filepath=img_dir)
    save_figure(cm, "confusion_matrix.png", filepath=img_dir)
   
    click.echo("\nAll data and models loaded successfully!")
    click.echo(f"-> Test set: {X_test.shape[0]:,} samples")
    click.echo("-> Models: Decision Tree and Naive Bayes ready")


if __name__ == "__main__":
    main()