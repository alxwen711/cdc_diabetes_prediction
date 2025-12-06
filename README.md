# cdc_diabetes_prediction

**Authors: Michael Eirikson, Raymond Wang, Alexander Wen**

## Summary

Basic data analysis on predicting diabetes based on health and lifestyle features following sound data scientific workflows as part of the main project for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

In this project we explored a decision tree model and naive bayes for predicting diabetes. After concluding that the decision tree is stronger in this context, We then utilized f2-score as our scoring function due to the context of our problem placing higher severity on false negatives rather than false positives for detecting diabetes.

We conclude that the decision tree model was the best performing of the models tested, correctly detected 8309 of 10604 positive cases (recall rate is about 78%). This result does come at a fairly significant cost in terms of false positives (precision rate is about 29%) with 20054 false positives. Depending on the actual cost of false positive this may need significant improvement to be a viable screening model.

The full report of our findings can be found [here](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb).

## Dependencies

- [Docker](https://www.docker.com/)

- [Docker image created for this analysis](https://hub.docker.com/r/meirikson/cdc_diabetes_prediction)

## Usage

### Setup

1. Clone this GitHub repo

### Analysis

1. Start Docker Desktop
2. In terminal navigate to the project root folder and run this command: `docker compose up`
3. In the terminal output for the above command look for a ULR beginning with `http://127.0.0.1:8888/lab?token=`
4. Copy the above URL in its entirety
5. Paste the URL into any web browser
6. Run the analysis by opening `diabetes_predition/notebooks/cdc_diabetes_prediction_report.ipynb` and clicking Run > Run All Cells

### Closing

To safely close the docker container

1. In terminal press `Crtl+C`
2. Once the container has stopped enter this terminal command to remove the container `docker-compose rm` type `y` to confirm

### Known Issues

On some apple silicon machines there is an issue with the kernel in jupyterlabs. The kernel may hang when started or when restarted or if sklearn functions use n_jobs>1.

If these are issues try:

- safely closing the container as described above
- restart the contrainer as described above
- don't click on anything except to navigate to the `cdc_diabetes_prediction_report.ipynb` notebook and click Run > Run All Cells

### Updating the environment and docker image

To update the environment and docker image follow these steps

1. Make sure you have a clean, current version of the envrionment by running

```bash
conda activate cdc_diabetes_prediction
conda env update --file environment.yml --prune
```

2. Install any new libraries
    - If conda fails to resolve dependencies try updating `environment.yml` and removing version numbers.
    - Note python must remain v 3.11 `python=3.11.6`
    - Note this is risky, different library version may cause issues builing the docker image

```bash
conda install <package>
```

3. Update `environment.yml` with

```bash
conda export --from-history > environment.yml
```

4. Create new `conda-linux-64.lock` file

```bash
conda-lock -k explicit --file environment.yml -p linux-64
```

5. Commit and push branch to remote repo.
    - The docker publish workflow will trigger and build and push a new docker image to DockerHub, then update the image tag in `docker-compose.yml`
    - Once the workflow is complete, pull the branch to locally get the updated `docker-comose.yml` file

6. Launch the new container with

```bash
docker-compose up
```

7. Within the terminal, run the following scripts to create all necessary files in the `results` folder:
```
python scripts/01-download_extract.py

python scripts/02-clean_transform_data.py # edit with exact command(s)

python scripts/03-split_preprocess_data.py # edit with exact command(s)

python scripts/04-EDA.py -c saveallcharts -p results/figures
python scripts/04-EDA.py -c describe -p results/tables

python scripts/05-model_fitting.py # edit with exact command(s)

python scripts/06-model_evaluation.py # edit with exact command(s)
```

8. Then use the following command to generate the report in both HTML and PDF format:
```
quarto render reports/cdc_diabetes_prediction_report.qmd
```

If the PDF format results in an error similar to [Issue #48](https://github.com/alxwen711/cdc_diabetes_prediction/issues/48#issue-3700114411), run this command first to ensure the fonts can be loaded properly:

```
quarto install tinytex
```


## References and Acknowledgements

The dataset utilized is the CDC Behavioural Risk Factor Surveillance System (BRFSS) 2015 Diabetes Health Indicators dataset (UCI ID 891), containing 253,680 survey responses with 21 health-related features and a binary diabetes outcome (0 = no diabetes/pre-diabetes, 1 = diabetes). A cleaned version of this dataset has been prepared by Aex Teboul and be accessed through Kaggle under the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data). Lastly, our project makes use of the `ucimlrepo` library to access the dataset more easily, further documentation for this tool is located at [https://github.com/uci-ml-repo/ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo).

The full list of references for the project can be found in the `References` section of the [CDC Diabetes Prediction report](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb)

## [License](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/LICENSE)

The [CDC Diabetes Prediction report](https://github.com/alxwen711/cdc_diabetes_prediction/blob/main/notebooks/cdc_diabetes_prediction_report.ipynb) is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) and the software code in this repository is licensed under the [MIT license](https://opensource.org/license/mit) For all usage/remixes of this project pease provide attribution and link to this webpage.
