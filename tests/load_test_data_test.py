import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_test_data import load_test_data

# Fixtures for creating sample data files
@pytest.fixture
def sample_X_test():
    """Create sample test features DataFrame"""
    return pd.DataFrame({
        "age": [45, 67, 23, 56],
        "bmi": [25.3, 30.1, 22.5, 28.7],
        "blood_pressure":  [120, 140, 110, 130],
        "glucose": [85, 110, 75, 95]
    })

# Fixture for creating sample y_test data
@pytest.fixture
def sample_y_test():
    """Create sample test labels DataFrame"""
    return pd.DataFrame({
        "diabetes": [0, 1, 0, 1]
    })

# --- Test for Missing Files ---
def test_missing_x_file_only(tmp_path, sample_y_test):
    """Test FileNotFoundError when only X_test file is missing"""
    x_path = tmp_path / "nonexistent_X_test.csv"
    y_path = tmp_path / "diabetes_y_test.csv"
    
    # Create only y_test file
    sample_y_test.to_csv(y_path, index=False)
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_test_data(x_path=str(x_path), y_path=str(y_path))
    
    # Assert the exact error message
    assert f"Test features file not found: {x_path}" in str(exc_info.value)


def test_missing_y_file_only(tmp_path, sample_X_test):
    """Test FileNotFoundError when only y_test file is missing"""
    x_path = tmp_path / "diabetes_X_test.csv"
    y_path = tmp_path / "nonexistent_y_test.csv"
    
    # Create only X_test file
    sample_X_test.to_csv(x_path, index=False)
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_test_data(x_path=str(x_path), y_path=str(y_path))
    
    # Assert the exact error message
    assert f"Test labels file not found: {y_path}" in str(exc_info.value)


def test_missing_both_files(tmp_path):
    """Test FileNotFoundError when both files are missing - X is checked first"""
    x_path = tmp_path / "nonexistent_X_test.csv"
    y_path = tmp_path / "nonexistent_y_test.csv"
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_test_data(x_path=str(x_path), y_path=str(y_path))
    
    # X_test is checked first, so that error should be raised
    assert f"Test features file not found: {x_path}" in str(exc_info.value)

# --- Test for Successful Load ---
def test_successful_load_returns_dataframe_and_series_and_logs_count(tmp_path, sample_X_test, sample_y_test, capsys):
    """When both files exist and are valid, function returns (DataFrame, Series) and echoes sample count."""
    x_path = tmp_path / "diabetes_X_test.csv"
    y_path = tmp_path / "diabetes_y_test.csv"

    # write CSV files without index
    sample_X_test.to_csv(x_path, index=False)
    sample_y_test.to_csv(y_path, index=False)

    X_test, y_test = load_test_data(x_path=str(x_path), y_path=str(y_path))

    # Types
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

    # Shapes & values
    assert X_test.shape == (4, 4)
    assert y_test.shape == (4,)
    assert list(y_test) == [0, 1, 0, 1]

    # Check click.echo output contains sample count
    captured = capsys.readouterr()
    assert "Loaded test set: 4 samples" in captured.out

# --- Test for KeyError on Missing 'diabetes' Column ---
def test_missing_diabetes_column_raises_keyerror(tmp_path, sample_X_test):
    """If y CSV does not contain 'diabetes' column, raise KeyError with explanatory message."""
    x_path = tmp_path / "diabetes_X_test.csv"
    y_path = tmp_path / "bad_y_test.csv"

    # write X and a y file that lacks 'diabetes'
    sample_X_test.to_csv(x_path, index=False)
    pd.DataFrame({"not_diabetes": [0, 1, 0, 1]}).to_csv(y_path, index=False)

    with pytest.raises(KeyError) as exc_info:
        load_test_data(x_path=str(x_path), y_path=str(y_path))

    # Expect message to indicate missing 'diabetes' column
    assert "Target column 'diabetes' not found" in str(exc_info.value)

# --- Test for Handling Index Columns ---
def test_load_with_index_column_in_y_csv(tmp_path, sample_X_test, sample_y_test):
    """CSV files that include an explicit index column (Unnamed: 0) should still load and return correct series."""
    x_path = tmp_path / "diabetes_X_test.csv"
    y_path = tmp_path / "diabetes_y_test.csv"

    # Save y with index column to simulate files that were saved with index=True
    sample_X_test.to_csv(x_path, index=True)   # also include index in X to simulate real-world files
    sample_y_test.to_csv(y_path, index=True)

    X_test, y_test = load_test_data(x_path=str(x_path), y_path=str(y_path))

    # The function should still find the 'diabetes' column and return correct series
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert list(y_test) == [0, 1, 0, 1]
