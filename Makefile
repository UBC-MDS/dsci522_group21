
python src/01__load_and_preprocess_data.py --input-data data/raw/bank-full.csv --output-data-dir data/processed

python src/02__eda_images_output.py --train data/processed/train_df.csv

python src/03__correlation_of_numerical_features.py --train data/processed/train_df.csv

python src/04__evaluate_models.py --x-train data/processed/X_train.csv --y-train data/processed/y_train.csv --output-cv-results data/processed/cv_results.csv --output-model-pipes data/processed/model_pipes.pkl



python src/06__fit_model.py --model-pipes data/processed/model_pipes.pkl
