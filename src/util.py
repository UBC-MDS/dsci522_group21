import pandas as pd
import altair as alt


def plot_eda(data, numerical_cols=[], categorical_cols=[]):
    """
    Returns two EDA plots, one for numerical columns and another for categorical columns.

    Parameters:
    ----------
    data : pd.DataFrame
        Input data that is used for plotting
    numerical_cols : list
        List of numerical columns that are to be plotted
    categorical_cols : list
        List of categorical columns that are to be plotted

    Returns:
    ----------
    (alt.Chart, alt.Chart)
        Two EDA plots, one for numerical columns and another for categorical columns
    """
    numerical_plot = None
    categorical_plot = None

    if len(categorical_cols) != 0:
        categorical_plot = alt.Chart(data).mark_bar().encode(
            x="count()",
            y=alt.Y(alt.repeat()).type("nominal")
        ).repeat(
            categorical_cols, columns=3
        )

    if len(numerical_cols) != 0:
        numerical_plot = alt.Chart(data).mark_bar().encode(
            x=alt.X(alt.repeat()).type("quantitative").bin(maxbins=40),
            y="count()"
        ).repeat(
            numerical_cols, columns=3
        )

    return numerical_plot, categorical_plot
