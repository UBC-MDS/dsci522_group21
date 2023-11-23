import pandas as pd
import altair as alt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def load_data_and_split(file_path, test_size=0.2, random_state=42, delimiter=None):
    
    """
    Load data from a specified file and split it into training and testing datasets.

    This function is designed to work with a variety of file formats including CSV, TXT, Excel (XLS/XLSX), and JSON.
    It reads the data into a pandas DataFrame and splits it into two separate DataFrames for training and testing purposes.
    
    The function assumes that the input file is structured with features and an optional target variable. If a delimiter
    is not specified for CSV/TXT files, the function defaults to using a comma (','). For Excel and JSON files, the delimiter
    parameter is ignored.

    Parameters:
    ----------
    file_path : str
        The path to the data file.
    
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split. Should be between 0.0 and 1.0.
    
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    
    delimiter : str, optional (default=None)
        The delimiter used in the file if it's a CSV or TXT. If None, defaults to ',' for CSV.

    Returns:
    -------
    train_df : DataFrame
        The training dataset, containing a subset of rows from the original data file.
    
    test_df : DataFrame
        The testing dataset, containing the remaining rows not included in the training dataset.

    Raises:
    ------
    ValueError
        If the file type is not supported (not in CSV, TXT, Excel, or JSON formats).

    Examples:
    --------
    To split a CSV file with a comma as a delimiter:

    >>> train_df, test_df = load_data_and_split('data/bank-full.csv', test_size=0.25, random_state=123)

    To split a CSV file with a semi-colon as a delimiter:

    >>> train_df, test_df = load_data_and_split('data/bank-full.csv', test_size=0.25, random_state=123, delimiter=';')

    Note:
    -----
    The file path provided to the function must be accessible from the current working directory.
    """
    
    # Determine the file type
    _, file_extension = os.path.splitext(file_path)
    
    # Load the data according to the file type
    if file_extension in ['.csv', '.txt']:
        data = pd.read_csv(file_path, delimiter=delimiter)
    elif file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(file_path)
    elif file_extension == '.json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type!")
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    return train_df, test_df

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

def plot_logistic_regression_feature_importance(fitted_lr_pipe, head=None, precision=3, cmap="PiYG", vmin=None, vmax=None):
    """
    Plots the feature importance for a fitted logistic regression model.

    Parameters:
    ----------
    fitted_lr_pipe : sklearn.pipeline.Pipeline
        Fitted pipeline containing a ColumnTransformer and LogisticRegression as its components.
    head : int, default=None
        Number of top features to display in the plot. If None, all features are displayed.
    precision : int, default=3
        Number of decimal places to round the coefficients.
    cmap : str, default="PiYG"
        Colormap for the background gradient in the plot.
    vmin : float, default=None
        Minimum value for the colormap scale.
    vmax : float, default=None
        Maximum value for the colormap scale.

    Returns:
    ----------
    pandas.io.formats.style.Styler
        Styled data frame containing the sorted feature importance with columns: 'feature' and 'coef'.

    Raises:
    ----------
    TypeError
        If `fitted_lr_pipe` does not have exactly 2 components: ColumnTransformer and LogisticRegression.
        If the 1st component in `fitted_lr_pipe` is not a ColumnTransformer.
        If the 2nd component in `fitted_lr_pipe` is not a LogisticRegression.
        If ColumnTransformer has no Encoder.
        If LogisticRegression is not fitted.

    ValueError
        If the number of features does not match the number of coefficients.
    """

    if len(fitted_lr_pipe.named_steps) != 2:
        raise TypeError("`fitted_lr_pipe` is expected to have exactly two components: ColumnTransformer and LogisticRegression")
    
    ct, lr = fitted_lr_pipe.named_steps.values()
    
    if not isinstance(ct, ColumnTransformer):
        raise TypeError("1st component in the `fitted_lr_pipe` is expected to be a ColumnTransformer")
    if not isinstance(lr, LogisticRegression):
        raise TypeError("2nd component in the `fitted_lr_pipe` is expected to be a LogisticRegression")
    if len(ct.named_transformers_) == 0:
        raise TypeError("ColumnTransformer has no Encoder")
    try:
        coef = lr.coef_[0]
    except AttributeError as e:
        raise TypeError("LogisticRegression is not fitted (`fitted_lr_pipe.fit(X_train, y_train))")
        
    features = []
    for enc in ct.named_transformers_.values():
        features += enc.get_feature_names_out().tolist()
        
    if len(features) != len(coef):
        raise ValueError("The number of features does not match the number of coefficients")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'coef': coef.tolist(),
        'coef_abs': abs(coef).tolist()
    })

    feature_importance = (
        feature_importance
        .sort_values('coef_abs', ascending=False)
        .reset_index(drop=True)
        .drop(columns=["coef_abs"])
        .head(head)
        .style.format(
            precision=3)
        .background_gradient(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            axis=None)
    )

    return feature_importance

def plot_correlation_heatmap(data, method='pearson'):
    """
    Returns a correlation heatmap for the numerical columns of the DataFrame provided.
        Input data that for which the correlation heatmap is to be generated.
    method : str, Default='pearson'
        Method of correlation to be used. Options are 'pearson' or 'spearman'. 

    Returns:
    ----------
    alt.LayerChart
        An Altair LayerChart object displaying a correlation heatmap in addition to correlation values.

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
