import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_test_data import load_test_data


@pytest.fixture
def sample_X_test():
    """Create sample test features DataFrame"""
    return pd.DataFrame({
        "age": [45, 67, 23, 56],
        "bmi": [25.3, 30.1, 22.5, 28.7],
        "blood_pressure":  [120, 140, 110, 130],
        "glucose": [85, 110, 75, 95]
    })


@pytest.fixture
def sample_y_test():
    """Create sample test labels DataFrame"""
    return pd.DataFrame({
        "diabetes": [0, 1, 0, 1]
    })


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

