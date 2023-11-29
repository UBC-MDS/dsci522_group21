import os
import click
import pandas as pd
from sklearn.metrics import classification_report
from util import plot_logistic_regression_feature_importance
import pickle

@click.command()
@click.option("--model", help="path to the model file")
@click.option("--x_test", help="path to data X_test")
@click.option("--y_test", help="path to data y_test")
def main(model, x_test, y_test):
    X_test = pd.read_csv(x_test)
    y_test = pd.read_csv(y_test)["y"]
    model = pickle.load(open(model, "rb"))

    df_report = pd.DataFrame(classification_report(y_test, model.predict(X_test), output_dict=True)).T
    df_report[["precision", "recall", "f1-score"]] = df_report[["precision", "recall", "f1-score"]].round(2)
    df_report["support"] = df_report["support"].astype(int)

    Fig_8 = plot_logistic_regression_feature_importance(model, head=5, precision=3, cmap="PiYG", vmin=None, vmax=None)

    df_report.to_csv("data/processed/classification_report.csv")
    Fig_8.data.to_csv("data/processed/feature_importance.csv")

if __name__ == "__main__":
    main()
