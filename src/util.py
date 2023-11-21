import logging
import pandas as pd
import altair as alt


def eda_plots(data: pd.DataFrame, numerical_cols: list = [], categorical_cols: list = []):
    """
    Returns two EDA plots, one for numerical columns and another for categorical columns.

    Parameters:
    ----------
    data : pd.DataFrame
        input data that is used for plotting
    numerical_cols : list
        list of numerical columns that are to be plotted
    categorical_cols : list
        list of categorical columns that are to be plotted

    Returns:
    ----------
    (alt.Chart, alt.Chart)
        two EDA plots, one for numerical columns and another for categorical columns
    """
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
