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
    Fig_4 = plot_correlation_heatmap(train_df)

    # Generate the pair-wise scatter plot for 'pdays' and 'previous' (Figure 5)
    Fig_5 = alt.Chart(train_df).mark_point().encode(
        x=alt.X("pdays", title='Days (pdays)'),
        y=alt.Y("previous", title='Interactions (previous)').scale(domain=(0, 50), clamp=True))

    # Format
    title_fig_4 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 4 - Pearson Correlation', 
        align='center',
        dx=-75,
        fontSize=13
    ).properties(width=400)
    
    title_fig_5 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 5 - Recency (pdays) vs Intensity (previous)', 
        align='center',
        dx=-45,
        fontSize=13
    ).properties(width=400)

    Fig_4_with_title = alt.vconcat(title_fig_4, Fig_4).resolve_scale(color='independent')
    Fig_5_with_title = alt.vconcat(title_fig_5, Fig_5).resolve_scale(color='independent')

    # Save the individual figures
    Fig_4_with_title.save("img/correlation_heatmap.png")
    Fig_5_with_title.save("img/pdays_vs_previous_scatter.png")

#    # Combine figures and save
#    all_figures = alt.hconcat(Fig_4_with_title, Fig_5_with_title).resolve_legend(color='independent')
#    all_figures.save("img/fig4_and_fig5_with_title.png")

if __name__ == "__main__":
    main()
