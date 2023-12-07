# Identifying the Top Three Predictors of Term Deposit Subscriptions Analysis

  - Authors: Jerry Yu, John Shiu, Sophia Zhao, Zeily Garcia
  - Contributors: 

This repository contains the code and analysis for the project "Identifying the Top Three Predictors of Term Deposit Subscriptions". Our team has explored a dataset from a Portuguese bank's marketing campaigns to understand what factors contribute to clients' decisions to subscribe to term deposits.

## About

This repository documents our analysis of factors influencing whether clients subscribe to a term deposit at a Portuguese bank. We used a dataset with 45,211 client interactions and 17 variables. We applied logistic regression and a decision tree classifier to identify the top three factors that predict subscription likelihood. We preprocessed the data by handling missing values, encoding categorical variables, and standardizing numerical ones for robust analysis. Our exploratory data analysis used visualizations to understand feature distributions and correlations. Our model evaluation prioritized precision due to class imbalance and minimizing false positives. Logistic regression performed slightly better than the decision tree. Our analysis highlighted prior campaign outcomes, timing of client contacts, and call duration as key indicators of subscription likelihood. These insights can inform marketing strategies for banking products and future research.

## Insights and Future Directions

This analysis identified three important factors influencing client subscriptions to term deposits at a Portuguese bank: prior campaign success, seasonal trends (higher in March, lower in January), and longer call durations. Logistic Regression performed better than the Decision Tree model, but both faced challenges due to class imbalance. To improve, we can use balanced sampling, explore Decision Tree insights, and try advanced models. This understanding helps banks refine marketing strategies. Future research could involve more complex models, additional data, and exploring economic factors.

## Report

The complete report can be found [here](https://ubc-mds.github.io/group21_top-three-predictors-of-term-deposit-subscriptions/term_deposit_report.html).

## Usage

There are two ways of using this repository: by creating our conda environment, or by using Docker.

### Conda Environment

To replicate the analysis:

1. Clone this repository.
   ```bash
   git@github.com:UBC-MDS/group21_top-three-predictors-of-term-deposit-subscriptions.git
   ```
2. Install the required dependencies. When running for the first time, please create conda environment by running this command:
   ```bash
   conda env create -f environment.yml -n 522
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
git clone git@github.com:UBC-MDS/group21_top-three-predictors-of-term-deposit-subscriptions.git
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

To run the analysis in jupter notebook, navigate to the `src/term_deposit_report.ipynb` notebook. Then, from the "Kernel" menu, select "Restart Kernel and Run All Cells...".  

Alternatively, to run in the terminal, execute the following commands from the project root:
```bash
# Load, preprocess, and split data into training and testing sets.
python src/01__load_and_preprocess_data.py \
    --input-data data/raw/bank-full.csv \
    --output-data-dir data/processed

# Generate and save exploratory data analysis (EDA) plots for training data.
python src/02__eda_images_output.py \
    --train data/processed/train_df.csv \
    --output-categorical img/job_types.png \
    --output-numerical img/previous_and_pdays.png 

# Explore correlation between numeric variables in training data and generate plots.
python src/03__correlation_of_numerical_features.py \
    --train data/processed/train_df.csv \
    --output-heatmap img/correlation_heatmap.png \
    --output-scatterplot img/pdays_vs_previous_scatter.png

# Evaluate machine learning models using cross-validation and save the results and model pipelines.
python src/04__evaluate_models.py \
    --x-train data/processed/X_train.csv \
    --y-train data/processed/y_train.csv \
    --output-cv-results data/processed/cv_results.csv \
    --output-model-pipes data/processed/model_pipes.pkl

# Process cross-validation results and save the mean values.
python src/05__extract_cross_validation_means.py \
    --cv-results data/processed/cv_results.csv \
    --output-cv-means data/processed/cv_means.csv

# Fit a specified Logistic Regression model and save the fitted model pipeline.
python src/06__fit_model.py \
    --x-train data/processed/X_train.csv \
    --y-train data/processed/y_train.csv \
    --model-pipes data/processed/model_pipes.pkl \
    --model-to-fit LogisticRegression \
    --output-model-pipe data/processed/logistic_regression.pkl

# Evaluate a fitted Logistic Regression model on test data and generate feature importance information.
python src/07__evaluate_model_and_feature_importance.py \
    --model data/processed/logistic_regression.pkl \
    --x-test data/processed/X_test.csv \
    --y-test data/processed/y_test.csv \
    --output-eval-report data/processed/classification_report.csv \
    --output-feat-importance data/processed/feature_importance.csv
```
### Using Makefile

To build the project report, clone this repo with dependencies, open your terminal and navigate to the project root directory, then run the following command:
```bash
make all
```
To clean up generated files and data, open your terminal and navigate to the project root directory, then run the following command:
```bash
make clean
```

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
- click=8.1.7
- jupyter-book=0.15.1

```

## File Consumption Order

1. Datasets: `bank-full.csv`, `bank-names.csv`, `bank.csv`, `bank-additional-full.csv`, `bank-additional-names.csv`, `bank-additional.csv`
2. Scripts: `term_deposit_report.ipynb`, `term_deposit_full_analysis.ipynb`
3. Report: `term_deposit_report.html`


## License

This project materials are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
The code contained within this repository is licensed under the MIT license. 
See the [license file](LICENSE) for more information.
 

# References

- Harris, C.R. et al., 2020. Array programming with NumPy. Nature, 585, pp.357–362.
- Varada Kolhatkar and Joel Ostblom. Dsci_573_feat-model-select_students repository. 2023. URL: https://github.ubc.ca/MDS-2023-24/DSCI_573_feat-model-select_students/blob/fc08e3246ff07f0425942d1b97429d5d0ebce933/environment.yaml.
- S. Moro, P. Rita, and P. Cortez. Bank Marketing. UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C5K306.
- F. Pedregosa and others. Scikit-learn: machine learning in python. Journal of machine learning research, 12:2825–2830, 2011.
- Guido Van Rossum and Fred L. Drake. Python 3 Reference Manual. CreateSpace, Scotts Valley, CA, 2009. ISBN 1441412697.
- Jacob VanderPlas, Brian E. Granger, Jeffrey Heer, Dominik Moritz, Kanit Wongsuphasawat, Arvind Satyanarayan, Eitan Lees, Ilia Timofeev, Ben Welsh, and Scott Sievert. Altair: interactive statistical visualizations for python. Journal of Open Source Software, 3(32):1057, 2018. URL: https://doi.org/10.21105/joss.01057, doi:10.21105/joss.01057.
- Wes McKinney. Data Structures for Statistical Computing in Python. In Stéfan van der Walt and Jarrod Millman, editors, Proceedings of the 9th Python in Science Conference, 56 – 61. 2010. doi:10.25080/Majora-92bf1922-00a.
