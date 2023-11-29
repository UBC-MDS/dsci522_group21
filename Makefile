
python src/01__load_and_preprocess_data.py --input-data data/raw/bank-full.csv --output-data-dir data/processed

python src/02__eda_images_output.py --train data/processed/train_df.csv

python src/03__correlation_of_numerical_features.py --train data/processed/train_df.csv
