import pytest
import numpy as np
import pandas as pd
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util import *

np.random.seed(123)
test_data_eda = pd.DataFrame({
    "cate1": ["a", "a", "b", "b", "b", "c", "a", "b", "a", "c"],
    "cate2": ["c", "c", "b", "b", "b", "a", "c", "b", "c", "a"],
    "num1": np.random.normal(size=10),
    "num2": np.random.normal(size=10)
})

def test_eda_none():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1"])
    num_chart_3, cate_chart_3 = eda_plots(test_data_eda)

    assert num_chart_1 is None, "Numerical chart should be None without input for numerical columns."
    assert cate_chart_1 is not None, "Categorical chart should not be None."
    assert num_chart_2 is not None, "Numerical chart should not be None."
    assert cate_chart_2 is None, "Categorical chart should be None without input for numerical columns."
    assert num_chart_3 is None, "Numerical chart should be None without input for numerical columns."
    assert cate_chart_3 is None, "Categorical chart should be None without input for numerical columns."

def test_eda_chart_type():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1"])
    num_chart_3, cate_chart_3 = eda_plots(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_4, cate_chart_4 = eda_plots(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert isinstance(cate_chart_1, alt.RepeatChart), "Incorrect data type for categorical chart."
    assert isinstance(num_chart_2, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(num_chart_3, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(cate_chart_3, alt.RepeatChart), "Incorrect data type for categorical chart."
    assert isinstance(num_chart_4, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(cate_chart_4, alt.RepeatChart), "Incorrect data type for categorical chart."

def test_eda_repeat():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert num_chart_1.to_dict()["repeat"] == ["num1"], "Wrong columns were being repeated for num_chart_1."
    assert cate_chart_1.to_dict()["repeat"] == ["cate1"], "Wrong columns were being repeated for cate_chart_1."
    assert num_chart_2.to_dict()["repeat"] == ["num1", "num2"], "Wrong columns were being repeated for num_chart_2."
    assert cate_chart_2.to_dict()["repeat"] == ["cate1", "cate2"], "Wrong columns were being repeated for cate_chart_2."

def test_eda_mark():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert num_chart_1.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert "bin" in num_chart_1.to_dict()["spec"]["encoding"]["x"].keys(), "Should be a histogram."
    assert cate_chart_1.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert num_chart_2.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert "bin" in num_chart_2.to_dict()["spec"]["encoding"]["x"].keys(), "Should be a histogram."
    assert cate_chart_2.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."

def test_eda_data_type():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert num_chart_1.to_dict()["spec"]["encoding"]["x"]["type"] == "quantitative", "Wrong data type for x in numerical chart."
    assert num_chart_1.to_dict()["spec"]["encoding"]["y"]["aggregate"] == "count", "Wrong data type for y in numerical chart."
    assert num_chart_1.to_dict()["spec"]["encoding"]["y"]["type"] == "quantitative", "Wrong data type for y in numerical chart."
    assert cate_chart_1.to_dict()["spec"]["encoding"]["x"]["aggregate"] == "count", "Wrong data type for x in categorical chart."
    assert cate_chart_1.to_dict()["spec"]["encoding"]["x"]["type"] == "quantitative", "Wrong data type for x in categorical chart."
    assert cate_chart_1.to_dict()["spec"]["encoding"]["y"]["type"] == "nominal", "Wrong data type for y in categorical chart."
    assert num_chart_2.to_dict()["spec"]["encoding"]["x"]["type"] == "quantitative", "Wrong data type for x in numerical chart."
    assert num_chart_2.to_dict()["spec"]["encoding"]["y"]["aggregate"] == "count", "Wrong data type for y in numerical chart."
    assert num_chart_2.to_dict()["spec"]["encoding"]["y"]["type"] == "quantitative", "Wrong data type for y in numerical chart."
    assert cate_chart_2.to_dict()["spec"]["encoding"]["x"]["aggregate"] == "count", "Wrong data type for x in categorical chart."
    assert cate_chart_2.to_dict()["spec"]["encoding"]["x"]["type"] == "quantitative", "Wrong data type for x in categorical chart."
    assert cate_chart_2.to_dict()["spec"]["encoding"]["y"]["type"] == "nominal", "Wrong data type for y in categorical chart."

def test_eda_dataset():
    num_chart_1, cate_chart_1 = eda_plots(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = eda_plots(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert test_data_eda.equals(pd.DataFrame(list(num_chart_1.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(cate_chart_1.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(num_chart_2.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(cate_chart_2.to_dict()["datasets"].values())[0])), "Chart is not using the right data."