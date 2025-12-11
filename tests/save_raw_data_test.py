import pandas as pd
import pytest
import sys
import os

# Import the save_raw_data function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_raw_data import save_raw_data

# basic use case
X_simple = pd.DataFrame({"x": [1,2,3]})
y_simple = pd.DataFrame(data = [4,5,6])
simple_df = pd.DataFrame({"x": [1,2,3], "y": [4,5,6]})

def test_simple():
    save_raw_data(X_simple, y_simple, filepath = "testdata/save_raw_data", filename = "simple.csv", label = "y")
    output_df = pd.read_csv("testdata/save_raw_data/simple.csv")
    pd.testing.assert_frame_equal(output_df,simple_df)

# X is not a DataFrame
X_notdf = "what"

def test_notX():
    with pytest.raises(TypeError):
        save_raw_data(X_notdf,y_simple)


# Y is not a DataFrame
y_notdf = pd.Series([5,4,3])

def test_noty():
    with pytest.raises(TypeError):
        save_raw_data(X_simple,y_notdf)


# Y consists of the wrong number of columns
y_threecol = pd.DataFrame(data = [[1,1,1],[2,2,2],[3,3,3]])

def test_not_one_col():
    with pytest.raises(ValueError):
        save_raw_data(X_simple,y_threecol)


# X and Y dimensions are incompatible

y_toolong = pd.DataFrame(data = [1,2,3,4])
y_tooshort = pd.DataFrame(data = [1,2])

def test_wrong_dims():
    with pytest.raises(IndexError):
        save_raw_data(X_simple, y_toolong, filepath = "testdata/save_raw_data", filename = "impossible.csv", label = "y")
    with pytest.raises(IndexError):
        save_raw_data(X_simple, y_tooshort, filepath = "testdata/save_raw_data", filename = "impossible.csv", label = "y")
    