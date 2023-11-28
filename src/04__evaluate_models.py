import os
import pickle
import click
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score


@click.command()
@click.option('--input-x-train', help='Path to the CSV file containing the features training (X_train) data.')
@click.option('--input-y-train', help='Path to the CSV file containing the target training (y_train) data.')
@click.option('--output-cv-results', default=None, help='Path to save cross-validation results as a CSV file. If None, results will not be saved. (Default: None)') 
@click.option('--output-model-pipes', default=None, help='Path to save model pipelines as a PKL file. If None, pipelines will not be saved. (Default: None)') 
def main(input_x_train, 
         input_y_train, 
         output_cv_results, 
         output_model_pipes):
    """ Preprocess features, and evaluate machine learning models using 5-fold cross-validation. """
    X_train = pd.read_csv(input_x_train, index_col=0)
    y_train = pd.read_csv(input_y_train, index_col=0)["y"]

    # Categorize different types of features
    categorical_feats = ["job", "marital", "default", "housing", "loan", "contact", "day", "month", "poutcome"]
    ordinal_feats = ["education"]
    numeric_feats = ["age", "balance", "duration", "campaign", "previous", "pdays"]

    education_levels = ["unknown", "primary", "secondary", "tertiary"]

    # Define preprocessor for each type of features
    preprocessor = make_column_transformer(
        (OneHotEncoder(sparse_output=False, drop="if_binary"), categorical_feats),
        (OrdinalEncoder(categories=[education_levels], dtype=int), ordinal_feats),
        (StandardScaler(), numeric_feats),
    )

    # Define model pipelines for each model to be evaluated
    model_pipes = {
        "Baseline": DummyClassifier(strategy="most_frequent", random_state=522),
        "DecisionTree": make_pipeline(preprocessor, DecisionTreeClassifier(max_depth=5, random_state=522)),
        "LogisticRegression": make_pipeline(preprocessor, LogisticRegression(max_iter=2000, random_state=522)),
    }

    # Define scoring metrics to be included in the cross-validation results
    mod_precision_score = make_scorer(precision_score, zero_division=0)
    classification_metrics = {
        "accuracy": "accuracy",
        "precision": mod_precision_score,
        "recall": "recall", 
    }

    # Perform 5-fold cross-validation for each model
    cross_val_results = {}
    for name, pipe in model_pipes.items():
        cross_val_results[name] = pd.DataFrame(
            cross_validate(
                pipe, 
                X_train, 
                y_train=="yes", 
                cv=5,
                return_train_score=True, 
                scoring=classification_metrics)
        ).agg(['mean', 'std']).round(3).T

    # Save results and model pipelines
    if output_cv_results is not None:
        pd.concat(cross_val_results, axis='columns').to_csv(output_cv_results)
    if output_model_pipes is not None:
        with open(output_model_pipes, 'wb') as file:  
            pickle.dump(model_pipes, file) 


if __name__ == '__main__':
    main()
