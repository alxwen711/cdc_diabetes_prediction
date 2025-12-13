import pandas as pd
import pytest
import sys
import os

# Import the generate_label function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.generate_label import generate_label

# simple use case
simple_df = pd.DataFrame({'feature_value': [1,2], 'diabetes': [3,4]})
simple_expected_df = pd.DataFrame({'feature_value': [1,2], 
                                   'diabetes': [3,4],
                                   'label': ['f = 1, d = 3', 'f = 2, d = 4']})

def test_simple():
    simple_df["label"] = simple_df.apply(generate_label, axis = 1)
    pd.testing.assert_frame_equal(simple_df,simple_expected_df)


# test arguments
complex_df = pd.DataFrame({"a": [True], "b": ["snth"]})
complex_expected_df = pd.DataFrame({"a": [True], "b": ["snth"], "c": ["e = True, f = snth"]})

def test_complex():
    complex_df["c"] = complex_df.apply(generate_label,
                                      feature_name = "a",
                                      feature_label = "e",
                                      value_name = "b",
                                      value_label = "f",
                                      axis = 1)
    pd.testing.assert_frame_equal(complex_df,complex_expected_df)


# edge case for one column
mono_df = pd.DataFrame({"number": [1,2,3,4,5]})
mono_expected_df = pd.DataFrame({"number": [1,2,3,4,5], 
                                 "mono": ["f = 1, d = 1", "f = 2, d = 2", "f = 3, d = 3", "f = 4, d = 4", "f = 5, d = 5"]})

def test_mono_column():
    mono_df["mono"] = mono_df.apply(generate_label, feature_name = "number", value_name = "number", axis = 1)
    pd.testing.assert_frame_equal(mono_df, mono_expected_df)


# column does not exist in dataframe error 
not_found_df = pd.DataFrame({"nope": ["NO","non","nada","0","nothing"]}) 

def test_column_not_found():
    with pytest.raises(KeyError):
        not_found_df["answer"] = not_found_df.apply(generate_label, feature_name = "yes", axis = 1)


# dataframe has no columns

empty_df = pd.DataFrame()

def test_empty_dataframe():
    with pytest.raises(ValueError):
        empty_df["aaaaa"] = empty_df.apply(generate_label, axis = 1)    

