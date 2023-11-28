import os
import sys
import pandas as pd
import altair as alt
from util import plot_eda
import click

@click.command()
@click.option("--train_path", help="path to the training data")

def main(train_path):
    train_df = pd.read_csv(train_path)
    alt.data_transformers.enable("vegafusion")

    _, categorical_plot = plot_eda(train_df, categorical_cols=['job'])
    Fig_1 = categorical_plot

    numerical_plot,_ = plot_eda(train_df, numerical_cols=['previous', 'pdays'])
    Fig_2_3 = numerical_plot

    title_fig_1 = alt.Chart({'values':[{}]}).mark_text(
        text='Figure 1 - Distribution of Job Types', 
        align='center',
        dx=-60,
        fontSize=13
    ).properties(width=400)

    Fig1_with_title = alt.vconcat(title_fig_1, Fig_1)

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

    Fig1_with_title.save("img/fig1_with_title.png")
    Fig_2_3_with_title.save("img/fig_2_3_with_title.png")

if __name__ == "__main__":
    main()