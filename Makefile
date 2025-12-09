.PHONY: all clean

all: reports/cdc_diabetes_prediction_report.html reports/cdc_diabetes_prediction_report.pdf

# Step 1: Download and save raw CDC diabetes data
data/raw/diabetes_raw.csv: scripts/01-download_extract.py
	python scripts/01-download_extract.py \
		--filepath=data/raw \
		--filename=diabetes_raw.csv \
		--label=diabetes

# Step 2: Clean, validate, split, and save train/test diabetes datasets
data/clean/diabetes_clean_train.csv data/clean/diabetes_clean_test.csv: scripts/02-clean_transform.py data/raw/diabetes_raw.csv
	python scripts/02-clean_transform.py \
		--file=data/raw/diabetes_raw.csv \
		--savefilepath=data/clean

# Step 3: Split into final processed X/y train/test files
data/processed/diabetes_X_train.csv data/processed/diabetes_y_train.csv data/processed/diabetes_X_test.csv data/processed/diabetes_y_test.csv: scripts/03-split_preprocess_data.py \
data/clean/diabetes_clean_train.csv \
data/clean/diabetes_clean_test.csv
	python scripts/03-split_preprocess_data.py \
		--clean-train=data/clean/diabetes_clean_train.csv \
		--clean-test=data/clean/diabetes_clean_test.csv \
		--output-dir=data/processed

# Step 4: EDA
results/figures/EDA_count.png results/figures/EDA_histogram.png results/figures/EDA_boxplot.png results/figures/EDA_correlation.png results/figures/EDA_binary.png: scripts/04-EDA.py \
data/processed/diabetes_X_train.csv \
data/processed/diabetes_y_train.csv
	python scripts/04-EDA.py --command saveallcharts --path results/figures

# Step 5: Fit models and save as pickle files
results/models/tree_model.pickle results/models/naive_bayes_model.pickle: scripts/05-model_fitting.py data/processed/diabetes_X_train.csv data/processed/diabetes_y_train.csv
	python scripts/05-model_fitting.py \
		--xfile=data/processed/diabetes_X_train.csv \
		--yfile=data/processed/diabetes_y_train.csv

# Step 6: Evaluate models and produce test set results
results/figures/model_performance_comparison.png results/figures/confusion_matrix.png results/tables/model_scores.csv: scripts/06-model_evaluation.py \
data/processed/diabetes_X_test.csv \
data/processed/diabetes_y_test.csv \
results/models/tree_model.pickle \
results/models/naive_bayes_model.pickle
	python scripts/06-model_evaluation.py \
		--x-test=data/processed/diabetes_X_test.csv \
		--y-test=data/processed/diabetes_y_test.csv \
		--model-dir=results/models \
		--img-dir=results/figures

# Step 7: Generate final report
reports/cdc_diabetes_prediction_report.html reports/cdc_diabetes_prediction_report.pdf: reports/cdc_diabetes_prediction_report.qmd \
results/figures/EDA_count.png \
results/figures/EDA_histogram.png \
results/figures/EDA_boxplot.png \
results/figures/EDA_correlation.png \
results/figures/EDA_binary.png \
results/figures/model_performance_comparison.png \
results/figures/confusion_matrix.png \
results/tables/model_scores.csv \
reports/references.bib
	quarto install tinytex --quiet --no-prompt || true
	quarto render reports/cdc_diabetes_prediction_report.qmd


clean:
	rm -rf data/raw/diabetes_raw.csv
	rm -rf data/clean/diabetes_clean_train.csv
	rm -rf data/clean/diabetes_clean_test.csv
	rm -rf data/processed/diabetes_X_train.csv
	rm -rf data/processed/diabetes_y_train.csv
	rm -rf data/processed/diabetes_X_test.csv
	rm -rf data/processed/diabetes_y_test.csv
	rm -rf results/figures/EDA_count.png
	rm -rf results/figures/EDA_histogram.png
	rm -rf results/figures/EDA_boxplot.png
	rm -rf results/figures/EDA_correlation.png
	rm -rf results/figures/EDA_binary.png
	rm -rf results/figures/model_performance_comparison.png
	rm -rf results/figures/confusion_matrix.png
	rm -rf results/tables/model_scores.csv
	rm -rf results/models/tree_model.pickle
	rm -rf results/models/naive_bayes_model.pickle
	rm -rf reports/cdc_diabetes_prediction_report.html
	rm -rf reports/cdc_diabetes_prediction_report.pdf