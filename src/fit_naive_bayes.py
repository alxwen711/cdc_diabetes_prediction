import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV

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