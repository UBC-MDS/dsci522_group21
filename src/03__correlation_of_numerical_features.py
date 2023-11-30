import click
import pandas as pd
import altair as alt
from util import plot_correlation_heatmap

@click.command()
@click.option("--train", help="Path to the training data CSV file.")

def main(train):
    """Explores the correlation between numeric variables in the specified training data and plots pair-wise scatter plot for the highly correlated variables."""

    # Read the training data
    train_df = pd.read_csv(train)

    # Enable altair vegafusion for data with more than 5000 rows
    alt.data_transformers.enable("vegafusion")

    # Generate the correlation heatmap (Figure 4)
    heatmap = plot_correlation_heatmap(train_df)

    # Generate the pair-wise scatter plot for 'pdays' and 'previous' (Figure 5)
    scatter = alt.Chart(train_df).mark_point().encode(
        x=alt.X("pdays", title='Days (pdays)'),
        y=alt.Y("previous", title='Interactions (previous)').scale(domain=(0, 50), clamp=True))

    # Save the individual figures
    heatmap.save("img/correlation_heatmap.png")
    scatter.save("img/pdays_vs_previous_scatter.png")


if __name__ == "__main__":
    main()
