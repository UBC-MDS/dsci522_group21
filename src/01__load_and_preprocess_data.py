import os
import click
import pandas as pd
from util import load_data_and_split

@click.command()
@click.option('--input-data', prompt='Path to input data file', help='Path to the input data CSV file.')
@click.option('--output-data-dir', prompt='Directory to save preprocessed data', help='Directory where preprocessed CSV data will be saved.')
def main(input_data, output_data_dir):
    """ Load and split the input data into training and testing sets, save the resulting data frames. """

    # load data and split into train and test set for X and target y
    train_df, test_df = load_data_and_split(input_data, delimiter=";")
    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
    X_test, y_test = test_df.drop(columns=["y"]), test_df["y"]

    # save data into csv
    train_df.to_csv(os.path.join(output_data_dir, "train_df.csv"))
    test_df.to_csv(os.path.join(output_data_dir, "test_df.csv"))
    X_train.to_csv(os.path.join(output_data_dir, "X_train.csv"))
    y_train.to_csv(os.path.join(output_data_dir, "y_train.csv"))
    X_test.to_csv(os.path.join(output_data_dir, "X_test.csv"))
    y_test.to_csv(os.path.join(output_data_dir, "y_test.csv"))

    train_df["y"].value_counts(normalize=True).to_csv(os.path.join(output_data_dir, 'y_train_distribution.csv'))

if __name__ == '__main__':
    main()
