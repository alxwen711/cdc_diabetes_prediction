.PHONY: all clean

all: data/clean/diabetes_clean_train.csv data/clean/diabetes_clean_test.csv

# Step 1: Download and save raw CDC diabetes data from UCI to CSV
data/raw/diabetes_raw.csv: scripts/01-download_extract.py
	python scripts/01-download_extract.py \
		--filepath=data/raw \
		--filename=diabetes_raw.csv \
		--label=diabetes

# Step 2: Clean, validate, split and save train/test diabetes datasets
data/clean/diabetes_clean_train.csv data/clean/diabetes_clean_test.csv: scripts/02-clean_transform.py data/raw/diabetes_raw.csv
	python scripts/02-clean_transform.py \
		--file=data/raw/diabetes_raw.csv \
		--savefilepath=data/clean

clean:
	rm -rf data/raw/diabetes_raw.csv
	rm -rf data/clean/diabetes_clean_train.csv
	rm -rf data/clean/diabetes_clean_test.csv