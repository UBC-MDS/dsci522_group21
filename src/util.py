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

    if len(categorical_cols) == 0:
        logging.info("No categorical columns were provided, and no EDA plot for categorical columns will be generated.")
    else:
        logging.info("Generating EDA plot for categorical columns.")
        categorical_plot = alt.Chart(data).mark_bar().encode(
            x=alt.X(alt.repeat()).type("quantitative").bin(maxbins=40),
            y="count()"
        ).repeat(
            numerical_cols, columns=3
        )

    return numerical_plot, categorical_plot
