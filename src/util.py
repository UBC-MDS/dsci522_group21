import logging
import pandas as pd
import altair as alt


def eda_plots(data: pd.DataFrame, numerical_cols: list = [], categorical_cols: list = []):
    numerical_plot = None
    categorical_plot = None

    if len(numerical_cols) == 0:
        logging.info("No numerical columns were provided, and no EDA plot for numerical columns will be generated.")
    else:
        logging.info("Generating EDA plot for numerical columns.")
        numerical_plot = alt.Chart(data).mark_bar().encode(
            x="count()",
            y=alt.Y(alt.repeat()).type("nominal")
        ).repeat(
            categorical_cols, columns=3
        )

