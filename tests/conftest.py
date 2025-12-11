import pytest
import shutil
import os

# Test cleanup code is adapted from https://github.com/ttimbers/breast-cancer-predictor/blob/3.0.0/tests/conftest.py.

@pytest.fixture(autouse=True, scope='session')
def remove_testdata_directory():
    # This code will run at the end of the pytest session
    yield

    try:
        shutil.rmtree("testdata")
    except FileNotFoundError:
        print("directory not removed")
        pass  # Directory doesn't exist, continue