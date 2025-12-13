# Test suite notes

## Running tests

To run the test suite navigate to the project root directory and run the command `pytest` or `pytest -v` for more verbose output.

The following test files in this folder correlate directly to the function files in the `src` folder:

| `src` file    | Testcase file located in `tests` |
| -------- | ------- |
| fit_naive_bayes.py  | fit_naive_bayes_test.py    |
| generate_label.py | generate_label_test.py     |
| save_raw_data.py    | save_raw_data_test.py    |
| load_test_data.py  |  load_test_data_test.py  |

## Test teardown

Automatic teardown of files and directories created by tests is managed by [conftest.py](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/tests/conftest.py).
