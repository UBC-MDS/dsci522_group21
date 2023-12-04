all : report/_build/html/index.html


data/processed/train_df.csv \
	data/processed/test_df.csv \
	data/processed/X_train.csv \
	data/processed/y_train.csv \
	data/processed/X_test.csv \
	data/processed/y_test.csv : data/raw/bank-full.csv src/01__load_and_preprocess_data.py
	python src/01__load_and_preprocess_data.py --input-data data/raw/bank-full.csv --output-data-dir data/processed

img/job_types.png img/previous_and_pdays.png : data/processed/train_df.csv src/02__eda_images_output.py
	python src/02__eda_images_output.py --train data/processed/train_df.csv --output-categorical img/job_types.png --output-numerical img/previous_and_pdays.png

img/correlation_heatmap.png img/pdays_vs_previous_scatter.png : data/processed/train_df.csv src/03__correlation_of_numerical_features.py
	python src/03__correlation_of_numerical_features.py --train data/processed/train_df.csv --output-heatmap img/correlation_heatmap.png --output-scatterplot img/pdays_vs_previous_scatter.png

data/processed/model_pipes.pkl data/processed/cv_results.csv : data/processed/X_train.csv data/processed/y_train.csv src/04__evaluate_models.py
	python src/04__evaluate_models.py --x-train data/processed/X_train.csv --y-train data/processed/y_train.csv --output-cv-results data/processed/cv_results.csv --output-model-pipes data/processed/model_pipes.pkl

data/processed/cv_means.csv : data/processed/cv_results.csv src/05__extract_cross_validation_means.py
	python src/05__extract_cross_validation_means.py --cv-results data/processed/cv_results.csv --output-cv-means data/processed/cv_means.csv

data/processed/logistic_regression.pkl : data/processed/model_pipes.pkl data/processed/y_train.csv src/06__fit_model.py
	python src/06__fit_model.py --x-train data/processed/X_train.csv --y-train data/processed/y_train.csv --model-pipes data/processed/model_pipes.pkl --model-to-fit LogisticRegression --output-model-pipe data/processed/logistic_regression.pkl

data/processed/classification_report.csv data/processed/feature_importance.csv : data/processed/logistic_regression.pkl data/processed/X_test.csv data/processed/y_test.csv src/07__evaluate_model_and_feature_importance.py
	python src/07__evaluate_model_and_feature_importance.py --model data/processed/logistic_regression.pkl --x-test data/processed/X_test.csv --y-test data/processed/y_test.csv --output-eval-report data/processed/classification_report.csv --output-feat-importance data/processed/feature_importance.csv

report/_build/html/index.html : \
	report/term_deposit_report.ipynb \
	report/_toc.yml \
	report/_config.yml \
	report/references.bib \
	img/job_types.png img/previous_and_pdays.png \
	img/correlation_heatmap.png img/pdays_vs_previous_scatter.png \
	data/processed/model_pipes.pkl data/processed/cv_results.csv \
	data/processed/cv_means.csv \
	data/processed/logistic_regression.pkl \
	data/processed/classification_report.csv data/processed/feature_importance.csv
	jupyter-book build report


clean :
	rm -f img/job_types.png img/previous_and_pdays.png
	rm -f img/correlation_heatmap.png img/pdays_vs_previous_scatter.png
	rm -f data/processed/model_pipes.pkl data/processed/cv_results.csv
	rm -f data/processed/cv_means.csv
	rm -f data/processed/logistic_regression.pkl
	rm -f data/processed/classification_report.csv data/processed/feature_importance.csv
	rm -rf report/_build
