import pandas as pd
import altair as alt

def correlation_heatmap(data, method='pearson'):
    """
    Returns a correlation heatmap for the numerical columns of the DataFrame provided.

    Parameters:
    ----------
    data : pd.DataFrame
        input data that for which the correlation heatmap is to be generated.
    method : str, Optional
        method of correlation to be used. Options are 'pearson' or 'spearman'. 
        Default is 'pearson'.

    Returns:
    ----------
    alt.LayerChart
        A Altair LayerChart object displaying a correlation heatmap in addition to correlation values.

    Raises:
    ----------
    ValueError
        If the method stated is not 'pearson' nor 'spearman'.
    """
    numerical_cols = data.select_dtypes(include=['int64', 'float64'])
    if method == 'pearson':
         corr_matrix = numerical_cols.corr(method='pearson')
         title = f'{method.capitalize()} Correlation'
    elif method == 'spearman':
         corr_matrix = numerical_cols.corr(method='spearman')
         title = f'{method.capitalize()} Correlation'
    else:
         raise ValueError("must use 'pearson' or 'spearman'")
    pear_corr_df = corr_matrix.unstack().reset_index()
    pear_corr_df.columns = ["num_variable_0", "num_variable_1", "correlation"]
    
    corr_heatmap = alt.Chart(pear_corr_df, title=title).mark_rect().encode(
        x=alt.X("num_variable_0:N", title="Numerical Variable"),
        y=alt.Y("num_variable_1:N", title="Numerical Variable"),
        color=alt.Color("correlation:Q")
    ).properties(
        width=250,
        height=250
    )
    text = alt.Chart(pear_corr_df).mark_text().encode(
        x=alt.X("num_variable_0:N").title("Numerical Variable"),
        y=alt.Y("num_variable_1:N").title("Numerical Variable"),
        text=alt.Text("correlation:Q", format=".2f")
    )
    return corr_heatmap + text