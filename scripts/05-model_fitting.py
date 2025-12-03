"""Fit decision tree and naive bayes models using training dataset.

This script with load the processed datasets with train and test data.
Then use grid searches to find good hyperparameters for both a
decision tree classifier and naive bayes classifier. A pickle file
of the decision tree model (the better model) will be saved.
"""

# imports
import pandas as pd


import os




f2_scorer = make_scorer(fbeta_score, beta=2)

###### TREE

def fit_decision_tree(a: int, b: int = 0) -> int:
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

    tree = DecisionTreeClassifier(random_state=522, class_weight='balanced')

    tree_params = {
        'max_depth': [6, 8, 10, 12, 14],
        'min_samples_leaf': [175, 200, 225, 250]
    }

    tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring=f2_scorer, n_jobs=1)
    tree_grid.fit(X_train, y_train)

    best_tree = tree_grid.best_estimator_
    


####### NB

preprocessor = make_column_transformer(
    (StandardScaler(), X_train.columns)
)

nb_pipe = make_pipeline(
    preprocessor,
    BernoulliNB()
)

nb_params = {'bernoullinb__alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}

knn_grid = GridSearchCV(nb_pipe, nb_params, cv=5, scoring=f2_scorer, n_jobs=1)
knn_grid.fit(X_train, y_train)

best_nb = knn_grid.best_estimator_
print("Best NB k:", knn_grid.best_params_)
print("Best CV f2-score:", knn_grid.best_score_.round(4))

