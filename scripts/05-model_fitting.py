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
import os

# functions
def fit_decision_tree(X_train: pd.DataFrame, y_train: pd.DataFrame) -> DecisionTreeClassifier:
    """Fit a decision tree model on training dataset to predict diabetes.
        
    Uses grid search to find good hyperparameters.

    Parameters
    ----------
    X_train : pd.Dataframe
        Training dataset of model features.
    y_train : pd.Datafram
        Target values for training dataset.
    
    Returns
    -------
    best_tree
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
    
def fit_naive_bayes(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Fit a bernoulli naive bayes model on training dataset to predict diabetes.
        
    Uses grid search to find good hyperparameters.
    
    Parameters
    ----------
    X_train : pd.Dataframe
        Training dataset of model features.
    y_train : pd.Datafram
        Target values for training dataset.
    
    Returns
    -------
    best_nb
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

