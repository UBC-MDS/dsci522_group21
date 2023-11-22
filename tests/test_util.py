import pytest
import numpy as np
import pandas as pd
import altair as alt
import sys
sys.path.append("../")

from src.util import *

def test_output_type():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    heatmap = correlation_heatmap(df)
    assert isinstance(heatmap, alt.LayerChart), "Output should be an alt.LayerChart object."

def test_invalid_method():
    data = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    try:
        heatmap = correlation_heatmap(data, method='invalid')
        assert False, "Invalid method should raise a ValueError."
    except ValueError:
        pass

def test_empty_dataframe():
    df = pd.DataFrame()
    heatmap = correlation_heatmap(df, method='spearman')
    assert heatmap.data.empty, "Function should return an empty heatmap for an empty DataFrame."

def test_no_numerical_columns():
    df = pd.DataFrame({
        'A': ['x', 'y', 'z'],
        'B': ['p', 'q', 'r'],
        'C': ['l', 'm', 'n']
    })
    heatmap = correlation_heatmap(df)
    assert heatmap.data.empty, "Function should return an empty heatmap for non numerical DataFrame."

    
def test_data_type_in_heatmap():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    heatmap = correlation_heatmap(df)
    for val in heatmap.data['correlation']:
        assert isinstance(val, (int, float)), "correlation values should be int or float data types."

def test_correct_correlation_calculation():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    heatmap = correlation_heatmap(df)
    expected_corr = df.corr(method='pearson').stack().reset_index(name='correlation')
    actual_corr = heatmap.data
    assert expected_corr['correlation'].equals(actual_corr['correlation']), "correlation values are not being calculated correctly."

def test_heatmap_labels():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    heatmap = correlation_heatmap(df)
    x_labels = set(heatmap.data['num_variable_0'].unique())
    y_labels = set(heatmap.data['num_variable_1'].unique())
    expected_labels = {'A', 'B'}
    
    assert x_labels == expected_labels, "X axis should match DataFrame columns."
    assert y_labels == expected_labels, "y axis should match DataFrame columns."