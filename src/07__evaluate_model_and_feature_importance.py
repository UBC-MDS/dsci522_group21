import click
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from util import plot_logistic_regression_feature_importance

@click.command()
@click.option("--model", help="Path to the fitted Logistic Regression model pipeline PKL file.")
@click.option("--x-test", help="Path to the CSV file containing the features testing (X_test) data.")
@click.option("--y-test", help="Path to the CSV file containing the target testing (y_test) data.")
def main(model, x_test, y_test):
    """ Evaluate classification report on a fitted Logistic Regression model on the test data, and output the feature importance. """
 
    # Read the test data
    X_test = pd.read_csv(x_test)
    y_test = pd.read_csv(y_test)["y"]

    # Load the fitted Logistic Regression model
    model = pickle.load(open(model, "rb"))

    # Generate and save a classification report
    df_report = pd.DataFrame(classification_report(y_test, model.predict(X_test), output_dict=True)).T
    df_report[["precision", "recall", "f1-score"]] = df_report[["precision", "recall", "f1-score"]].round(2)
    df_report["support"] = df_report["support"].astype(int)
    df_report.to_csv("data/processed/classification_report.csv")

    # Generate and save the feature importance plot
    Fig_8 = plot_logistic_regression_feature_importance(model, head=5, precision=3, cmap="PiYG", vmin=None, vmax=None)
    Fig_8.data.to_csv("data/processed/feature_importance.csv")

if __name__ == "__main__":
    main()
