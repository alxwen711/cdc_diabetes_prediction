# cdc_diabetes_prediction

**Authors: Michael Eirikson, Raymond Wang, Alexander Wen**

Basic data analysis on predicting the likelihood of diabetes given various lifestyle factors following sound data scientific workflows as part of the main project for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

In this project we explored a decision tree model and naive bayes for predicting diabetes. After concluding that the decision tree is stronger in this context, We then utilized f2-score as our scoring function due to the context of our problem placing higher severity on false negatives rather than false positives for detecting diabetes.

We conclude that the decision tree model was the best performing of the models tested, correctly detected 8283 of 10604 positive cases (recall rate is about 78%). This result does come at a fairly significant cost in terms of false positives (precision rate is about 30%) with 19650 false positives. Depending on the actual cost of false positive this may need significant improvement to be a viable screening model.

The full report of our findings can be found [here](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb).


## Dependencies

The dependencies for this project are everything listed in [environment.yml](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/environment.yml). The environment can be setup from this yml file by using conda (version 25.3.1 or higher) and conda-lock (version 3.0.4).

If conda-lock is not in the environment, run the following command:

```
conda install conda-lock=3.0.4
```

## Initial Setup

If this is the first time running this project, run the following from the repository root to setup the virtual environment:

```
conda-lock install --name cdc_diabetes_prediction conda-lock.yml
```

## Usage

Assuming the initial setup is complete, setup the initial environment with the following command.

```
conda activate cdc_diabetes_prediction
```

Then open up Jupyter Lab with the following command:

```
jupyter lab
```

Navigate to `cdc_diabetes_prediction_analysis.ipynb` in Jupyter Lab. Select the Python Kernel and then under the `Kernel` menu, click `Restart Kernel and Run All Cells`.

## References and Acknowledgements

The dataset utilized is the CDC Behavioural Risk Factor Surveillance System (BRFSS) 2015 Diabetes Health Indicators dataset (UCI ID 891), containing 253,680 survey responses with 21 health-related features and a binary diabetes outcome (0 = no diabetes/pre-diabetes, 1 = diabetes). A cleaned version of this dataset has been prepared by Aex Teboul and be accessed through Kaggle under the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data). Lastly, our project makes use of the `ucimlrepo` library to access the dataset more easily, further documentation for this tool is located at [https://github.com/uci-ml-repo/ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo).

The full list of references for the project can be found in the `References` section of the [CDC Diabetes Prediction report](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb)


## [License](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/LICENSE)

The [CDC Diabetes Prediction report](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb) is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) and the software code in this repository is licensed under the [MIT license](https://opensource.org/license/mit) For all usage/remixes of this project pease provide attribution and link to this webpage.
