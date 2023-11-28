import click
import pickle
import pandas as pd


@click.command()
@click.option('--input-x-train', help='Path to the CSV file containing the features training (X_train) data.')
@click.option('--input-y-train', help='Path to the CSV file containing the target training (y_train) data.')
@click.option('--input-model-pipes', help='Path to the PKL file containing the saved model pipelines.') 
@click.option('--model-to-fit', default='LogisticRegression', help='Name of the model to fit. Choose from ["Baseline", "DecisionTree", "LogisticRegression"]. (Default: LogisticRegression)') 
@click.option('--output-model-pipe', help='Path to save the fitted model pipeline as a PKL file.') 
def main(input_x_train,
         input_y_train,
         input_model_pipes,
         model_to_fit,
         output_model_pipe):

    """ Fit a specified machine learning model. """

    # Read data and load the saved model pipelines
    X_train = pd.read_csv(input_x_train, index_col=0)
    y_train = pd.read_csv(input_y_train, index_col=0)["y"]
    with open(input_model_pipes,'rb') as file:
        model_pipes = pickle.load(file)

    # Select the specified model to fit
    model_pipe = model_pipes[model_to_fit]
    model_pipe.fit(X_train, y_train)

    # Save the fitted model pipeline
    with open(output_model_pipe, 'wb') as file:
        pickle.dump(model_pipe, file)


if __name__ == '__main__':
    main()
