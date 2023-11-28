# load_and_split
# output train_df, test_df, X_train, y_train, X_test, y_test.csv
# ouput train_df into png
# output train_df['y'].value_counts into png

# arguments: 
# --input data
# --ouput_dir
# --print-train-df-into-image
# --print-y-train-into-image
# --print-X-train-into-image

import os
import click
import dataframe_image as dfi
from util import load_data_and_split

@click.command()
@click.option('--input-data', prompt='Path to input data file', help='Path to the input data CSV file.')
@click.option('--output-data-dir', prompt='Directory to save preprocessed data', help='Directory where preprocessed CSV data will be saved.')
@click.option('--output-img-dir', default=None, help='Directory to save output images.')
@click.option('--print_train_df_head_into_png', default=False, help='Print the head of the training DataFrame into a PNG image.')
@click.option('--print_X_train_head_into_png', default=False, help='Print the head of the features (X_train) into a PNG image.')
@click.option('--print_y_train_dist_into_png', default=False, help='Print the distribution of the target (y_train) into a PNG image.')
def main(input_data,
         output_data_dir,
         output_img_dir,
         print_train_df_head_into_png,
         print_X_train_head_into_png,
         print_y_train_dist_into_png):

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

    # print data frames into png
    if output_img_dir:
        if print_train_df_head_into_png:
            train_df.head().dfi.export(os.path.join(output_img_dir, 'train_df_head.png'))
        if print_X_train_head_into_png:
            X_train.head().dfi.export(os.path.join(output_img_dir, 'X_train_head.png'))
        if print_y_train_dist_into_png:
            pd.DataFrame(y_train.value_counts(normalize=True)).dfi.export(os.path.join(output_img_dir, 'y_train_dist.png'))
