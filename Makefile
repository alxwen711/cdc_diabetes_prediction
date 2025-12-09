.PHONY: all clean

all: data/raw/diabetes_raw.csv

# Download and save raw CDC diabetes data from UCI to CSV
data/raw/diabetes_raw.csv : scripts/01-download_extract.py
	python scripts/01-download_extract.py \
		--filepath=data/raw \
		--filename=diabetes_raw.csv \
		--label=diabetes

clean:
	rm -rf data/raw/diabetes_raw.csv