import pytest
import numpy as np
import pandas as pd
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.util import *

# Test data
simple_numeric_df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9]})

categorical_df = pd.DataFrame({
    'A': ['x', 'y', 'z'],
    'B': ['p', 'q', 'r'],
    'C': ['l', 'm', 'n']})

empty_df = pd.DataFrame()

# Test for correct return type
def test_output_type():
    heatmap = correlation_heatmap(simple_numeric_df)
    assert isinstance(heatmap, alt.LayerChart), "Output should be an alt.LayerChart object."

# Test for raising of error if method is invalid
def test_invalid_method():
    try:
        heatmap = correlation_heatmap(simple_numeric_df, method='invalid')
        assert False, "Invalid method should raise a ValueError."
    except ValueError:
        pass

# Test for empty dataframe input
def test_empty_dataframe():
    heatmap = correlation_heatmap(empty_df, method='spearman')
    assert heatmap.data.empty, "Function should return an empty heatmap for an empty DataFrame."

# Test for categorical columns input 
def test_no_numerical_columns():
    heatmap = correlation_heatmap(categorical_df)
    assert heatmap.data.empty, "Function should return an empty heatmap for non numerical DataFrame."

# Test for correct type for calculated values
def test_data_type_in_heatmap():
    heatmap = correlation_heatmap(simple_numeric_df)
    for val in heatmap.data['correlation']:
        assert isinstance(val, (int, float)), "correlation values should be int or float data types."

# Test for correct calculation
def test_correct_correlation_calculation():
    heatmap = correlation_heatmap(simple_numeric_df)
    expected_corr = simple_numeric_df.corr(method='pearson').stack().reset_index(name='correlation')
    actual_corr = heatmap.data
    assert expected_corr['correlation'].equals(actual_corr['correlation']), "correlation values are not being calculated correctly."

# Test for correct labels
def test_heatmap_labels():
    heatmap = correlation_heatmap(simple_numeric_df)
    x_labels = set(heatmap.data['num_variable_0'].unique())
    y_labels = set(heatmap.data['num_variable_1'].unique())
    expected_labels = {'A', 'B', 'C'}
    assert x_labels == expected_labels, "X axis should match DataFrame columns."
    assert y_labels == expected_labels, "y axis should match DataFrame columns."