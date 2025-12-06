"""Fit decision tree and naive bayes models using training dataset.

This script with load the processed datasets with train and test data.
Then use grid searches to find good hyperparameters for both a
decision tree classifier and naive bayes classifier. A pickle file
of the decision tree model (the better model) will be saved.
"""

# imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
import pickle
import os
import click

# functions
def load_training_data(X_file: str, y_file) -> tuple[pd.DataFrame, pd.Series]:
    """Load X_train and y_train csv files to pandas dataframes.
        
    Parameters
    ----------
    X_file : str
        Path and filename of csv file with X train data.
    y_file : str
        Path and filename of csv file with y train data.
    
    Returns
    -------
    pd.Dataframe
        X_train dataframe.
    pd.Series
        y_train series.
    
    """

    X_train = pd.read_csv(X_file)
    y_train = pd.read_csv(y_file)["diabetes"]
    return X_train, y_train

def fit_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Fit a decision tree model on training dataset to predict diabetes.
        
    Uses grid search to find good hyperparameters.

    Parameters
    ----------
    X_train : pd.Dataframe
        Training dataset of model features.
    y_train : pd.Series
        Target values for training dataset.
    
    Returns
    -------
    sklearn.tree.DecisionTreeClassifier
        The best decision free from grid search.
    
    """

    tree = DecisionTreeClassifier(random_state=522, class_weight='balanced')

    tree_params = {
        'max_depth': [6, 8, 10, 12, 14],
        'min_samples_leaf': [175, 200, 225, 250]
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)

    tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring=f2_scorer, n_jobs=1)
    tree_grid.fit(X_train, y_train)

    best_tree = tree_grid.best_estimator_

    return best_tree
    
def fit_naive_bayes(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a bernoulli naive bayes model on training dataset to predict diabetes.
        
    Uses grid search to find good hyperparameters.
    
    Parameters
    ----------
    X_train : pd.Dataframe
        Training dataset of model features.
    y_train : pd.Series
        Target values for training dataset.
    
    Returns
    -------
    sklearn.naive_bayes.BernoulliNB
        The best naive bayes model from grid search.
    
    """
    preprocessor = make_column_transformer(
        (StandardScaler(), X_train.columns)
    )

    nb_pipe = make_pipeline(
        preprocessor,
        BernoulliNB()
    )

    nb_params = {'bernoullinb__alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}

    f2_scorer = make_scorer(fbeta_score, beta=2)

    nb_grid = GridSearchCV(nb_pipe, nb_params, cv=5, scoring=f2_scorer, n_jobs=1)
    nb_grid.fit(X_train, y_train)

    best_nb = nb_grid.best_estimator_

    return best_nb

def pickle_models(model, file_name: str) -> None:
    """Save sklearn model as a pickle file given the file path and name.
    
    Parameters
    ----------
    model : a sklearn model
        The model to pickle
    file_path : str
        The file path to directory to save in, starting from the project root.
    filt_name : str
        The name to give the pickle file.

    Returns
    -------
    None
    
    """

    path = "results/models/"

    # check/create model data folder
    if not os.path.exists(path):
        os.makedirs(path)

    # save model
    path_file_name = path + file_name
    with open(path_file_name, "wb") as f:
        pickle.dump(model, f)


@click.command()
@click.option(
    "--xfile", 
    type = str, 
    default = "data/processed/diabetes_X_train.csv", 
    help = "Directory and file name of X train csv file."
)
@click.option(
    "--yfile", 
    type = str, 
    default = "data/processed/diabetes_y_train.csv", 
    help = "Directory and file name of y train csv file."
)
def main(xfile: str, yfile: str):
    X_train, y_train = load_training_data(xfile, yfile)
    print("Searching for best hyperparameters for decision tree classifier...")
    best_dt = fit_decision_tree(X_train, y_train)
    print("Searching for best hyperparameters for naive bayes classifier...")
    best_nb = fit_naive_bayes(X_train, y_train)
    pickle_models(best_dt, "tree_model.pickle")
    pickle_models(best_nb, "naive_bayes_model.pickle")

if __name__ == "__main__":
    main()