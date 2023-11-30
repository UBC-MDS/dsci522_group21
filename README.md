# Identifying the Top Three Predictors of Term Deposit Subscriptions Analysis

  - Authors: Jerry Yu, John Shiu, Sophia Zhao, Zeily Garcia
  - Contributors: 

This repository contains the code and analysis for the project "Identifying the Top Three Predictors of Term Deposit Subscriptions". Our team has explored a dataset from a Portuguese bank's marketing campaigns to understand what factors contribute to clients' decisions to subscribe to term deposits.

## About

This repository documents our analysis on the determinants of term deposit subscriptions within a Portuguese bank, harnessing a dataset that tracks 45,211 client interactions across 17 distinct variables. We have considered logistic regression and decision tree classifier to unearth the top three principal factors that predict a client's propensity to subscribe to a term deposit. The data preprocessing involves handling of missing entries, encoding of categorical variables, and standardization of numerical variables to prepare for robust analysis.

The crux of our exploratory data analysis was the strategic use of visualizations to unravel the nuances in feature distributions and inter-feature correlations. Our model evaluation was meticulously tailored to emphasize precision, a decision dictated by the inherent class imbalance present within the dataset as well as minimizing the false positive rate. In this rigorous analytical process, logistic regression emerged as a marginally more precise model compared to the decision tree classifier.

Significantly, the analysis culminated in pinpointing the outcome of prior marketing campaigns, the timing of client contact within the year, and the duration of the calls as pivotal indicators of subscription likelihood. These insights not only shed light on the client's decision-making dynamics but also carve out potential avenues for further scholarly inquiry and practical application in marketing strategies for banking products.

## Insights and Future Directions

The report's findings are instrumental for banking institutions to comprehend and predict customer behavior concerning term deposit subscriptions. This predictive understanding is essential for refining marketing approaches and enhancing the efficiency of future campaigns. We believe that these insights can serve as a cornerstone for further research, potentially exploring more sophisticated analytical models and integrating additional datasets to delve into the observed seasonal patterns and other underlying phenomena influencing client decisions.

## Report

The complete report can be found [here](https://htmlpreview.github.io/?https://github.com/UBC-MDS/dsci522_group21/blob/main/src/term_deposit_report.html).

## Usage

There are two ways of using this repository: by creating our conda environment, or by using Docker.

### Conda Environment

To replicate the analysis:

1. Clone this repository.
   ```bash
   git clone https://github.com/UBC-MDS/dsci522_group21.git
   ```
2. Install the required dependencies. When running for the first time, please create conda environment by running this command:
   ```bash
   conda env create -f environment.yml
   ```
3. Run the following command to activate the installed environment:
   ```bash
   conda activate 522
   ```
4. Launch Jupyter Lab by running `jupyter lab` and navigate to the `src/term_deposit_report.ipynb` notebook. Then, from the "Kernel" menu, select "Restart Kernel and Run All Cells...".
   ```bash
   Jupyter Lab
   ```

### Docker

There are two ways to run the analysis in a Docker container. However, one always needs to first clone the repo:

```bash
git clone git@github.com:UBC-MDS/dsci522_group21.git
```

#### (method 1) build/pull image and run
In the first method, one needs to obtain our Docker image in one of two ways:

1. In the root folder of the repository, run `docker build --tag <your_image_name> .`, or,
2. In the command line, run `docker pull johnshiu/dsci522_group21:main`. You should now have the image, and it is called `johnshiu/dsci522_group21`.

To run use the image, run:

```bash
docker run --rm -p 8888:8888 -v $(pwd):/home/jovyan <your_image_name>
```

One can then open jupyter lab using the link given by the command line output.

#### (method 2) docker-compose
In the second method, to create and run the container, one can run:

```bash
docker-compose up
```

This command will create an image if it does not already exist. After the user is finished with the analysis, press `Ctrl + C` on the keyboard on the command window, and then run `docker-compose down` to shut down the container.

#### run analysis
After running one of the above methods, in the terminal, copy the URL that starts with `http://127.0.0.1:8888/lab?token=` (for example, see the highlighted text in the terminal below), and paste it into your browser.
<img src="img/docker_jupyter_lab_url.png">

To run the analysis, navigate to the `src/term_deposit_report.ipynb` notebook. Then, from the "Kernel" menu, select "Restart Kernel and Run All Cells...". 

### Run Unit Tests

In the root folder of the repository, run:

```bash
pytest 
```


## Dependencies

```raw
- python=3.11
- ipython=8.17.2
- ipykernel=6.26.0
- jupyterlab=4.0.9
- matplotlib=3.8.2
- pandas=2.1.3
- scikit-learn=1.3.2
- altair=5.1.2
- vl-convert-python=1.1.0
- vegafusion=1.4.5
- vegafusion-jupyter=1.4.5
- pytest=7.4.3

```

## File Consumption Order

1. Datasets: `bank-full.csv`, `bank-names.csv`, `bank.csv`, `bank-additional-full.csv`, `bank-additional-names.csv`, `bank-additional.csv`
2. Scripts: `term_deposit_report.ipynb`, `term_deposit_full_analysis.ipynb`
3. Report: `term_deposit_report.html`

## Script Usage

### `01_load_and_preprocess_data.py`

This script is designed to load, preprocess, and split your input data into training and testing sets. It also provides options to save these sets into CSV files and, optionally, to export certain data visualizations as PNG images.

#### Arguments

- **`--input-data <path>`**: Path to the input data CSV file. This is a required parameter.
- **`--output-data-dir <directory>`**: Directory where preprocessed CSV data will be saved. This is a required parameter.
- **`--output-img-dir <directory>`**: Optional. Directory to save output images. If not specified, no images will be saved.
- **`--print-train-df-head-into-png`**: Optional. If set, the script will print the head of the training DataFrame into a PNG image.
- **`--print-x-train-head-into-png`**: Optional. If set, the script will print the head of the features (X_train) into a PNG image.
- **`--print-y-train-dist-into-png`**: Optional. If set, the script will print the distribution of the target (y_train) into a PNG image.

#### Usage 

To run the script with the minimum required parameters:

```bash
python 01_load_and_preprocess_data.py --input-data "path/to/input.csv" --output-data-dir "path/to/output/dir"
```

To export data visualizations as PNG images:
```bash
python 01_load_and_preprocess_data.py --input-data "path/to/input.csv" --output-data-dir "path/to/output/dir" --output-img-dir "path/to/img/dir" --print-train-df-head-into-png --print-x-train-head-into-png --print-y-train-dist-into-png
```

### `02_eda_images_output.py`

This script is specifically designed for generating and saving exploratory data analysis (EDA) plots for training data. It reads the training data from a specified CSV file, creates visualizations to understand the data better, and saves these visualizations as images.

#### Arguments

- **`--train <path>`**: This argument specifies the path to the training data CSV file. It is mandatory to provide this path for the script to function.

#### Usage 

Run the script:

```bash
python 02_eda_images_output.py --train "path/to/training_data.csv"
```

### `03_correlation_of_numerical_features.py`

This script explores the correlation between numeric variables in the specified training data. It generates a correlation heatmap and a pair-wise scatter plot for highly correlated variables, providing insights into the relationships between numerical features in your dataset.

#### Arguments

- **`--train <path>`**: Path to the training data CSV file. This is a required parameter.

#### Usage 

To run the script:

```bash
python 03_correlation_of_numerical_features.py --train "path/to/training_data.csv"
```

### `04_evaluate_models.py`

This script is designed for preprocessing features and evaluating multiple machine learning models using 5-fold cross-validation. It reads training data, applies appropriate feature transformations, and assesses different models based on accuracy, precision, and recall.

#### Arguments

- **`--x-train <path>`**: Path to the CSV file containing the features training (X_train) data. This is a required parameter.
- **`--y-train <path>`**: Path to the CSV file containing the target training (y_train) data. This is a required parameter.
- **`--output-cv-results <path>`**: Optional. Path to save cross-validation results as a CSV file. If not specified, results will not be saved.
- **`--output-model-pipes <path>`**: Optional. Path to save model pipelines as a PKL file. If not specified, pipelines will not be saved.

#### Usage 

To run the script with the required arguments:
```bash
python 04_evaluate_models.py --x-train "path/to/X_train.csv" --y-train "path/to/y_train.csv"
```

To save cross-validation results and model pipelines:
```bash
python 04_evaluate_models.py --x-train "path/to/X_train.csv" --y-train "path/to/y_train.csv" --output-cv-results "path/to/cv_results.csv" --output-model-pipes "path/to/model_pipelines.pkl"
```

### `05_cross_validation_result_graph.py`

This script processes the cross-validation results from a CSV file and extracts key metrics to be saved in a separate file. It's primarily designed to facilitate the analysis and visualization of model performance metrics.

#### Arguments

- **`--file-path <path>`**: Optional. Path to the `cv_results.csv` file. Defaults to `../data/processed/cv_results.csv` if not specified.
- **`--output-path <path>`**: Optional. Path for saving the processed results as a CSV file. Defaults to `../data/processed/Fig_6.csv` if not specified.

#### Usage 

To run the script with default file paths:
```bash
python 05_cross_validation_result_graph.py
```

To specify custom paths for the input and output files:
```bash
python 05_cross_validation_result_graph.py --file-path "path/to/cv_results.csv" --output-path "path/to/output/Fig_6.csv"
```

### `06_fit_model.py`

This script is tailored for fitting a specified machine learning model pipeline using training data. It reads the training data and a pre-saved model pipeline, fits the chosen model, and then saves the fitted model for future use or deployment.

#### Arguments

- **`--x-train <path>`**: Path to the CSV file containing the features training (X_train) data. This is a required parameter.
- **`--y-train <path>`**: Path to the CSV file containing the target training (y_train) data. This is a required parameter.
- **`--model-pipes <path>`**: Path to the PKL file containing the saved model pipelines. This is a required parameter.
- **`--model-to-fit <model_name>`**: Optional. Name of the model to fit. Choose from `["Baseline", "DecisionTree", "LogisticRegression"]`. Defaults to `LogisticRegression` if not specified.
- **`--output-model-pipe <path>`**: Path to save the fitted model pipeline as a PKL file. This is a required parameter.

#### Usage 

To fit a specific model and save the fitted pipeline:

```bash
python 06_fit_model.py --x-train "path/to/X_train.csv" --y-train "path/to/y_train.csv" --model-pipes "path/to/model_pipelines.pkl" --model-to-fit "DecisionTree" --output-model-pipe "path/to/fitted_model.pkl"
```
### `07_evaluate_model_and_feature_importance.py`

This script is designed for evaluating a fitted Logistic Regression model on test data. It generates a classification report and outputs a visualization of feature importance based on the model's coefficients.

#### Arguments

- **`--model <path>`**: Path to the fitted Logistic Regression model pipeline PKL file. This is a required parameter.
- **`--x-test <path>`**: Path to the CSV file containing the features testing (X_test) data. This is a required parameter.
- **`--y-test <path>`**: Path to the CSV file containing the target testing (y_test) data. This is a required parameter.

#### Usage Example

To evaluate the model and generate feature importance information:

```bash
python 07_evaluate_model_and_feature_importance.py --model "path/to/fitted_model.pkl" --x-test "path/to/X_test.csv" --y-test "path/to/y_test.csv"
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.  

# References

- Harris, C.R. et al., 2020. Array programming with NumPy. Nature, 585, pp.357–362.
- Moro,S., Rita,P., and Cortez,P., 2012. Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
- Ostblom,J., 2023. environment.yaml. DSCI_573_feat-model-select_students Repository. https://github.ubc.ca/MDS-2023-24/DSCI_573_feat-model-select_students/blob/fc08e3246ff07f0425942d1b97429d5d0ebce933/environment.yaml.
- Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825–2830.
- Timbers,T. , Ostblom,J., and Lee,M., 2023. Breast Cancer Predictor Report. GitHub repository, https://github.com/ttimbers/breast_cancer_predictor_py/blob/0.0.1/src/breast_cancer_predictor_report.ipynb.
- Van Rossum, Guido, and Fred L. Drake. 2009. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.
- VanderPlas, J. et al., 2018. Altair: Interactive statistical visualizations for python. Journal of open source software, 3(32), p.1057.
