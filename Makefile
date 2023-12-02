
python src/01__load_and_preprocess_data.py --input-data data/raw/bank-full.csv --output-data-dir data/processed

python src/02__eda_images_output.py --train data/processed/train_df.csv --output-categorical img/job_types.png --output-numerical img/previous_and_pdays.png

python src/03__correlation_of_numerical_features.py --train data/processed/train_df.csv --output-heatmap img/correlation_heatmap.png --output-scatterplot img/pdays_vs_previous_scatter.png

python src/04__evaluate_models.py --x-train data/processed/X_train.csv --y-train data/processed/y_train.csv --output-cv-results data/processed/cv_results.csv --output-model-pipes data/processed/model_pipes.pkl

python src/05__extract_cross_validation_means.py --cv-results data/processed/cv_results.csv --output-cv-means data/processed/cv_means.csv

python src/06__fit_model.py --x-train data/processed/X_train.csv --y-train data/processed/y_train.csv --model-pipes data/processed/model_pipes.pkl --model-to-fit LogisticRegression --output-model-pipe data/processed/logistic_regression.pkl

python src/07__evaluate_model_and_feature_importance.py --model data/processed/logistic_regression.pkl --x-test data/processed/X_test.csv --y-test data/processed/y_test.csv --output-eval-report data/processed/classification_report.csv --output-feat-importance data/processed/feature_importance.csv
