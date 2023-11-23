import os
import sys
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
import pandas as pd
import altair as alt
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

# Import functions from src/util.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # to run pytest at the root of the project
from src.util import plot_logistic_regression_feature_importance, plot_correlation_heatmap

# Test data
empty_df = pd.DataFrame()

test_X = pd.DataFrame({
    "cate1": ["a", "a", "b", "b", "b", "c", "a", "b", "a", "c"],
    "cate2": ["c", "c", "b", "b", "b", "a", "c", "b", "c", "a"],
    "num": [0.41 , -0.08, -0.46, -0.4, -2.15, -0.29,  0.04 , -0.46,  0.58,  0.72],
    "ord": [1, 2, 3, 1, 2, 3, 1, 2, 3, 3]
})
test_y = pd.Series(["no", "no", "yes", "no", "no", "no", "yes", "yes", "no", "no"])
test_preprocessor = make_column_transformer(
    (StandardScaler(), ["num"]),
    (OneHotEncoder(sparse_output=False, drop="if_binary"), ["cate1", "cate2"]),
    (OrdinalEncoder(categories=[[1, 2, 3]], dtype=int), ["ord"])
)
test_lr_pipe = make_pipeline(test_preprocessor, LogisticRegression())
test_lr_pipe.fit(test_X, test_y)
test_head = 8

n_features = sum([len(enc.get_feature_names_out().tolist()) for enc in test_lr_pipe.steps[0][1].named_transformers_.values()])


# Test for correct return type of `plot_logistic_regression_feature_importance`
def test_fi_return_type():
    output = plot_logistic_regression_feature_importance(test_lr_pipe)
    output_head = plot_logistic_regression_feature_importance(test_lr_pipe, head=test_head)
    n_features = sum([len(enc.get_feature_names_out().tolist()) for enc in test_lr_pipe.steps[0][1].named_transformers_.values()])

    assert isinstance(output, pd.io.formats.style.Styler), f"Output should be a Styler (current: {type(output)})."

# Test for the output of `plot_logistic_regression_feature_importance`
def test_fi_output():
    output = plot_logistic_regression_feature_importance(test_lr_pipe)
    output_head = plot_logistic_regression_feature_importance(test_lr_pipe, head=test_head)

    assert len(output.columns) == 2, "Output should have 2 columns only."
    assert output.data.shape[0] == n_features, "Number of rows should equal to number of features when `head` is not specified."
    assert output_head.data.shape[0] == test_head, "Number of rows should equal to specified `head`."
    assert output.data.iloc[:,0].dtype == np.dtype('O'), "First column must be a feature string vector."
    assert output.data.iloc[:,1].dtype == np.dtype('float64'), "Second column must be a coefficient numeric vector."
    assert all(output.data.iloc[:,1].abs() == output.data.iloc[:,1].abs().sort_values(ascending=False)), "Output is not sorted by the absolute value of coefficients"

# Test for the error exceptions of `plot_logistic_regression_feature_importance` 
def test_fi_errors():
    with pytest.raises(TypeError) as excinfo:
        plot_logistic_regression_feature_importance(
            make_pipeline(LogisticRegression())
        )
    assert str(excinfo.value) == "`fitted_lr_pipe` is expected to have exactly two components: ColumnTransformer and LogisticRegression" 

    with pytest.raises(TypeError) as excinfo:
        plot_logistic_regression_feature_importance(
            make_pipeline(OneHotEncoder(), LogisticRegression())
        )
    assert str(excinfo.value) == "1st component in the `fitted_lr_pipe` is expected to be a ColumnTransformer"

    with pytest.raises(TypeError) as excinfo:
        plot_logistic_regression_feature_importance(
            make_pipeline(test_preprocessor, Ridge())
        )
    assert str(excinfo.value) == "2nd component in the `fitted_lr_pipe` is expected to be a LogisticRegression"

    with pytest.raises(TypeError) as excinfo:
        plot_logistic_regression_feature_importance(
            make_pipeline(test_preprocessor, LogisticRegression())
        )
    assert str(excinfo.value) == "LogisticRegression is not fitted (`fitted_lr_pipe.fit(X_train, y_train))"

# Test for correct return type
def test_heatmap_output_type():
    heatmap = plot_correlation_heatmap(test_X)
    assert isinstance(heatmap, alt.LayerChart), "Output should be an alt.LayerChart object."

# Test for raising of error if method is invalid
def test_heatmap_invalid_method():
    try:
        heatmap = plot_correlation_heatmap(test_X, method='invalid')
        assert False, "Invalid method should raise a ValueError."
    except ValueError:
        pass

# Test for empty dataframe input
def test_heatmap_empty_dataframe():
    heatmap = plot_correlation_heatmap(empty_df, method='spearman')
    assert heatmap.data.empty, "Function should return an empty heatmap for an empty DataFrame."

# Test for categorical columns input 
def test_heatmap_no_numerical_columns():
    heatmap = plot_correlation_heatmap(test_X[['cate1', 'cate2']])
    assert heatmap.data.empty, "Function should return an empty heatmap for non numerical DataFrame."

# Test for correct type for calculated values
def test_heatmap_data_type_in_heatmap():
    heatmap = plot_correlation_heatmap(test_X)
    for val in heatmap.data['correlation']:
        assert isinstance(val, (int, float)), "correlation values should be int or float data types."

# Test for correct calculation
def test_heatmap_correct_correlation_calculation():
    heatmap = plot_correlation_heatmap(test_X)
    expected_corr = test_X.select_dtypes(include=['int64', 'float64']).corr(method='pearson').stack().reset_index(name='correlation')
    actual_corr = heatmap.data
    assert expected_corr['correlation'].equals(actual_corr['correlation']), "correlation values are not being calculated correctly."

# Test for correct labels
def test_heatmap_labels():
    heatmap = plot_correlation_heatmap(test_X)
    x_labels = set(heatmap.data['num_variable_0'].unique())
    y_labels = set(heatmap.data['num_variable_1'].unique())
    expected_labels = {'num', 'ord'}
    assert x_labels == expected_labels, "X axis should match DataFrame columns."
    assert y_labels == expected_labels, "y axis should match DataFrame columns."
