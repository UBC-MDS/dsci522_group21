import click
import pandas as pd
import altair as alt
from util import plot_eda

@click.command()
@click.option("--train", help="Path to the training data CSV file.")

def main(train):
    """ Generate and save exploratory data analysis (EDA) plots for the training data. """

    # Read the training data
    train_df = pd.read_csv(train)

    # Enable altair vegafusion for data with more than 5000 rows
    alt.data_transformers.enable("vegafusion")

    # Generate EDA plots
    _, categorical_plot = plot_eda(train_df, categorical_cols=['job'])
    numerical_plot,_ = plot_eda(train_df, numerical_cols=['previous', 'pdays'])

    # Save the generated plots
    categorical_plot.save("img/job_types.png")
    numerical_plot.save("img/previous_and_pdays.png")

if __name__ == "__main__":
    main()
