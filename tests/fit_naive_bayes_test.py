import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from src.fit_naive_bayes import fit_naive_bayes   # UPDATE this import to match project structure

# Fixture: required column names
@pytest.fixture
def feature_columns():
    return {
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
        'Income'
    }

# Fixture: small valid training dataset
@pytest.fixture
def small_training_data(feature_columns):
    X = pd.DataFrame(
        np.zeros((10, len(feature_columns))), 
        columns=sorted(feature_columns)
    )
    y = pd.Series([0, 1] * 5)
    return X, y

# ---- TYPE CHECKING TESTS ----

def test_xtrain_type_error():
    y = pd.Series([0])
    # Wrong type for X_train
    with pytest.raises(TypeError, match="X_train must be a pandas dataframe"):
        fit_naive_bayes(X_train=[1, 2, 3], y_train=y) # pyright: ignore[reportArgumentType]

def test_ytrain_type_error(feature_columns):
    X = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=sorted(feature_columns))
    # Wrong type for y_train
    with pytest.raises(TypeError, match="y_train must be a pandas series"):
        fit_naive_bayes(X_train=X, y_train=[0, 1]) # pyright: ignore[reportArgumentType]

# ---- EMPTY INPUT TESTS ----

def test_empty_xtrain(feature_columns):
    X = pd.DataFrame(columns=sorted(feature_columns))
    y = pd.Series([0])
    with pytest.raises(ValueError, match="X_train must contain at least one row."):
        fit_naive_bayes(X_train=X, y_train=y)

def test_empty_ytrain(feature_columns):
    X = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=sorted(feature_columns))
    y = pd.Series(dtype=int)
    with pytest.raises(ValueError, match="y_train must contain at least one row."):
        fit_naive_bayes(X_train=X, y_train=y)

# ---- COLUMN VALIDATION TEST ----

def test_wrong_columns(feature_columns):
    # Missing one required column
    bad_cols = sorted(list(feature_columns))[:-1]
    X = pd.DataFrame(np.zeros((5, len(bad_cols))), columns=bad_cols)
    y = pd.Series([0, 1, 0, 1, 0])

    with pytest.raises(ValueError, match="X_train should have features columns"):
        fit_naive_bayes(X_train=X, y_train=y)

# ---- SUCCESSFUL FIT TESTS ----

def test_fit_returns_pipeline_with_nb(small_training_data):
    X, y = small_training_data

    model = fit_naive_bayes(X, y)

    # should be a sklearn Pipeline from make_pipeline
    assert isinstance(model, Pipeline)
    # Last step should be BernoulliNB
    assert isinstance(model.steps[-1][1], BernoulliNB)

def test_model_can_predict_after_fit(small_training_data):
    X, y = small_training_data
    model = fit_naive_bayes(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)
    # Predictions should be binary {0,1}
    assert set(preds).issubset({0, 1})
