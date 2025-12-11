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
