all : report/_build/html/index.html


# Load, preprocess, and split data into training and testing sets.
data/processed/train_df.csv \
	data/processed/test_df.csv \
	data/processed/X_train.csv \
	data/processed/y_train.csv \
	data/processed/X_test.csv \
	data/processed/y_test.csv \
	data/processed/y_train_distribution.csv : data/raw/bank-full.csv src/01__load_and_preprocess_data.py
	python src/01__load_and_preprocess_data.py \
		--input-data data/raw/bank-full.csv \
		--output-data-dir data/processed

# Generate and save exploratory data analysis (EDA) plots for training data.
img/job_types.png img/previous_and_pdays.png : data/processed/train_df.csv src/02__eda_images_output.py
	python src/02__eda_images_output.py \
		--train data/processed/train_df.csv \
		--output-categorical img/job_types.png \
		--output-numerical img/previous_and_pdays.png

# Explore correlation between numeric variables in training data and generate plots.
img/correlation_heatmap.png img/pdays_vs_previous_scatter.png : data/processed/train_df.csv src/03__correlation_of_numerical_features.py
	python src/03__correlation_of_numerical_features.py \
		--train data/processed/train_df.csv \
		--output-heatmap img/correlation_heatmap.png \
		--output-scatterplot img/pdays_vs_previous_scatter.png

# Evaluate machine learning models using cross-validation and save the results and model pipelines.
data/processed/model_pipes.pkl data/processed/cv_results.csv : data/processed/X_train.csv data/processed/y_train.csv src/04__evaluate_models.py
	python src/04__evaluate_models.py \
		--x-train data/processed/X_train.csv \
		--y-train data/processed/y_train.csv \
		--output-cv-results data/processed/cv_results.csv \
		--output-model-pipes data/processed/model_pipes.pkl

# Process cross-validation results and save the mean values.
data/processed/cv_means.csv : data/processed/cv_results.csv src/05__extract_cross_validation_means.py
	python src/05__extract_cross_validation_means.py \
		--cv-results data/processed/cv_results.csv \
		--output-cv-means data/processed/cv_means.csv

# Fit a specified Logistic Regression model and save the fitted model pipeline.
data/processed/logistic_regression.pkl : data/processed/model_pipes.pkl data/processed/y_train.csv src/06__fit_model.py
	python src/06__fit_model.py \
		--x-train data/processed/X_train.csv \
		--y-train data/processed/y_train.csv \
		--model-pipes data/processed/model_pipes.pkl \
		--model-to-fit LogisticRegression \
		--output-model-pipe data/processed/logistic_regression.pkl

# Evaluate a fitted Logistic Regression model on test data and generate feature importance information.
data/processed/classification_report.csv data/processed/feature_importance.csv : data/processed/logistic_regression.pkl data/processed/X_test.csv data/processed/y_test.csv src/07__evaluate_model_and_feature_importance.py
	python src/07__evaluate_model_and_feature_importance.py \
		--model data/processed/logistic_regression.pkl \
		--x-test data/processed/X_test.csv \
		--y-test data/processed/y_test.csv \
		--output-eval-report data/processed/classification_report.csv \
		--output-feat-importance data/processed/feature_importance.csv

# Build report
report/_build/html/index.html : \
	report/_toc.yml \
	report/_config.yml \
	report/references.bib \
	report/term_deposit_report.ipynb \
	data/processed/train_df.csv \
	data/processed/y_train_distribution.csv \
	img/job_types.png \
	img/previous_and_pdays.png \
	img/correlation_heatmap.png \
	img/pdays_vs_previous_scatter.png \
	data/processed/cv_means.csv \
	data/processed/classification_report.csv \
	data/processed/feature_importance.csv
	jupyter-book build report


clean :
	rm -f data/processed/train_df.csv \
		data/processed/test_df.csv \
		data/processed/X_train.csv \
		data/processed/y_train.csv \
		data/processed/X_test.csv \
		data/processed/y_test.csv \
		data/processed/y_train_distribution.csv
	rm -f img/job_types.png img/previous_and_pdays.png
	rm -f img/correlation_heatmap.png img/pdays_vs_previous_scatter.png
	rm -f data/processed/model_pipes.pkl data/processed/cv_results.csv
	rm -f data/processed/cv_means.csv
	rm -f data/processed/logistic_regression.pkl
	rm -f data/processed/classification_report.csv data/processed/feature_importance.csv
	rm -rf report/_build
