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

    # Generate EDA plots for categorical columns (Figure 1)
    _, categorical_plot = plot_eda(train_df, categorical_cols=['job'])
    Fig_1 = categorical_plot

    # Generate EDA plots for numerical columns (Figure 2 and 3)
    numerical_plot,_ = plot_eda(train_df, numerical_cols=['previous', 'pdays'])
    Fig_2_3 = numerical_plot

    # Format for Figure 1
    title_fig_1 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 1 - Distribution of Job Types', 
        align='center',
        dx=-60,
        fontSize=13
    ).properties(width=400)

    Fig1_with_title = alt.vconcat(title_fig_1, Fig_1)

    # Format for Figure 2 and 3
    title_fig_2 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 2 - Number of Contacts Before this Campaign (previous)', 
        align='center',
        dx=-60,
        fontSize=13
    ).properties(width=400)

    title_fig_3 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 3 - Days Passed after Last Contact (pdays)', 
        align='center',
        dx=-60,
        fontSize=13
    ).properties(width=400)

    titles = alt.hconcat(title_fig_2, title_fig_3)
    Fig_2_3_with_title = alt.vconcat(titles, Fig_2_3)

    # Save the generated plots with titles
    Fig1_with_title.save("img/job_types.png")
    Fig_2_3_with_title.save("img/previous_and_pdays.png")

if __name__ == "__main__":
    main()
