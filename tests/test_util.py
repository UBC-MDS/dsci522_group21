import pytest
import numpy as np
import pandas as pd
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util import *

# Test data
test_X = pd.DataFrame({
    "cate1": ["a", "a", "b", "b", "b", "c", "a", "b", "a", "c"],
    "cate2": ["c", "c", "b", "b", "b", "a", "c", "b", "c", "a"],
    "num": [0.41 , -0.08, -0.46, -0.4, -2.15, -0.29,  0.04 , -0.46,  0.58,  0.72],
    "ord": [1, 2, 3, 1, 2, 3, 1, 2, 3, 3]})

empty_df = pd.DataFrame()

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