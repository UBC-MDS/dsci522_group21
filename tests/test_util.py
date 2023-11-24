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
from io import StringIO

# Import functions from src/util.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # to run pytest at the root of the project
from src.util import load_data_and_split, plot_eda, plot_logistic_regression_feature_importance, plot_correlation_heatmap

# Test data
# for load_data_and_split
data_csv = """feature1,feature2,label
1,2,0
3,4,1
5,6,0
7,8,1
9,10,0
1,2,0
3,4,1
5,6,0
7,8,1
9,10,0
"""
csv_file_path = 'data/sample_data.csv'
with open(csv_file_path, 'w') as file:
    file.write(data_csv)

txt_file_path = 'data/sample_data.txt'
with open(txt_file_path, 'w') as file:
    file.write(data_csv)

df = pd.read_csv(StringIO(data_csv))

excel_file_path = 'data/sample_data.xlsx'
df.to_excel(excel_file_path, index=False)

csv_semicolon_path = 'data/sample_data_semicolon.csv'
df.to_csv(csv_semicolon_path, index=False, sep=";")

csv_tab_path = 'data/sample_data_tab.csv'
df.to_csv(csv_tab_path, index=False, sep="\t")

json_path = 'data/sample_data.json'
df.to_json(json_path, index=False)

# for EDA plot
test_data_eda = pd.DataFrame({
    "cate1": ["a", "a", "b", "b", "b", "c", "a", "b", "a", "c"],
    "cate2": ["c", "c", "b", "b", "b", "a", "c", "b", "c", "a"],
    "num1": [-1.08 ,  0.997,  0.283, -1.51, -0.58, 1.65, -2.43, -0.43,  1.27, -0.87],
    "num2": [-0.68, -0.09,  1.49, -0.64, -0.44, -0.43,  2.21,  2.19,  1., 0.39]
})

# for correlation and feature importance
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


# Tests for cases when plot_eda should return None for at least one of the graphs
def test_eda_none():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1"])
    num_chart_3, cate_chart_3 = plot_eda(test_data_eda)

    assert num_chart_1 is None, "Numerical chart should be None without input for numerical columns."
    assert cate_chart_1 is not None, "Categorical chart should not be None."
    assert num_chart_2 is not None, "Numerical chart should not be None."
    assert cate_chart_2 is None, "Categorical chart should be None without input for numerical columns."
    assert num_chart_3 is None, "Numerical chart should be None without input for numerical columns."
    assert cate_chart_3 is None, "Categorical chart should be None without input for numerical columns."

# Tests for ensuring that the type of the charts is alt.RepeatChart
def test_eda_chart_type():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1"])
    num_chart_3, cate_chart_3 = plot_eda(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_4, cate_chart_4 = plot_eda(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert isinstance(cate_chart_1, alt.RepeatChart), "Incorrect data type for categorical chart."
    assert isinstance(num_chart_2, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(num_chart_3, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(cate_chart_3, alt.RepeatChart), "Incorrect data type for categorical chart."
    assert isinstance(num_chart_4, alt.RepeatChart), "Incorrect data type for numerical chart."
    assert isinstance(cate_chart_4, alt.RepeatChart), "Incorrect data type for categorical chart."

# Tests for ensuring that the correct columns are being repeated in the chart
def test_eda_repeat():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert num_chart_1.to_dict()["repeat"] == ["num1"], "Wrong columns were being repeated for num_chart_1."
    assert cate_chart_1.to_dict()["repeat"] == ["cate1"], "Wrong columns were being repeated for cate_chart_1."
    assert num_chart_2.to_dict()["repeat"] == ["num1", "num2"], "Wrong columns were being repeated for num_chart_2."
    assert cate_chart_2.to_dict()["repeat"] == ["cate1", "cate2"], "Wrong columns were being repeated for cate_chart_2."

# Tests for the correct mark type in the charts
def test_eda_mark():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert num_chart_1.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert "bin" in num_chart_1.to_dict()["spec"]["encoding"]["x"].keys(), "Should be a histogram."
    assert cate_chart_1.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert num_chart_2.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."
    assert "bin" in num_chart_2.to_dict()["spec"]["encoding"]["x"].keys(), "Should be a histogram."
    assert cate_chart_2.to_dict()["spec"]["mark"]["type"] == "bar", "Wrong mark type."

# Tests for the correct data type of x-axis and y-axis
def test_eda_data_type():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

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

# Tests for ensuring that the graph is using the correct dataset
def test_eda_dataset():
    num_chart_1, cate_chart_1 = plot_eda(test_data_eda, numerical_cols=["num1"], categorical_cols=["cate1"])
    num_chart_2, cate_chart_2 = plot_eda(test_data_eda, numerical_cols=["num1", "num2"], categorical_cols=["cate1", "cate2"])

    assert test_data_eda.equals(pd.DataFrame(list(num_chart_1.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(cate_chart_1.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(num_chart_2.to_dict()["datasets"].values())[0])), "Chart is not using the right data."
    assert test_data_eda.equals(pd.DataFrame(list(cate_chart_2.to_dict()["datasets"].values())[0])), "Chart is not using the right data."

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

# Test to ensure the split ratio is correct for csv file
def test_load_split_ratio_csv():
    train_df, test_df = load_data_and_split(csv_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The split ratio for the test set should be approximately 0.2."

# Test to ensure the function raises a ValueError for unsupported file types
def test_load_unsupported_file_type():
    with pytest.raises(ValueError):
        _, _ = load_data_and_split('/path/to/data.unsupported') 
    assert str(excinfo.value) == "Unsupported file type!", "A ValueError should be raised for unsupported file types."

# Test to ensure the function correctly uses the delimiter
def test_load_delimiter_handling():
    # This should pass as we are using the default delimiter which is a comma
    train_df, test_df = load_data_and_split(csv_file_path, delimiter=',')
    assert not train_df.empty and not test_df.empty, "DataFrames should not be empty when the correct delimiter is used."

# Test to ensure the split ratio is correct for excel file
def test_load_excel_data_loading():
    train_df, test_df = load_data_and_split(excel_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The split ratio for the test set should be approximately 0.2 for an Excel file."

# Test to ensure the function works for other common delimiters
def test_load_another_delim():
    train_df, test_df = load_data_and_split(csv_semicolon_path, delimiter=';')
    train_df_2, test_df_2 = load_data_and_split(csv_tab_path, delimiter='\t')
    assert not train_df.empty and not test_df.empty, "DataFrames should not be empty when the semicolon delimiter is used."
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The split ratio should be correct for semicolon delimiter."
    assert not train_df_2.empty and not test_df_2.empty, "DataFrames should not be empty when the tab delimiter is used."
    assert len(test_df_2) / (len(test_df_2) + len(train_df_2)) == pytest.approx(0.2, 0.01), "The split ratio should be correct for tab delimiter."

# Test to ensure the function works for text files
def test_load_txt():
    train_df, test_df = load_data_and_split(txt_file_path)
    assert not train_df.empty and not test_df.empty, "DataFrames should not be empty when loading from text files."
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The split ratio for the test set should be approximately 0.2 for text files."

# Test to ensure the function works for json files
def test_load_json():
    train_df, test_df = load_data_and_split(json_path)
    assert not train_df.empty and not test_df.empty, "DataFrames should not be empty when loading from JSON files."
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The split ratio for the test set should be approximately 0.2 for JSON files."

# Test to ensure the function works with default and non-default split size
def test_load_split_ratio():
    train_df, test_df = load_data_and_split(csv_file_path)
    train_df_2, test_df_2 = load_data_and_split(csv_file_path, test_size=0.3)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01), "The default split ratio for the test set should be approximately 0.2."
    assert len(test_df_2) / (len(test_df_2) + len(train_df_2)) == pytest.approx(0.3, 0.01), "The non-default split ratio for the test set should be approximately 0.3."
