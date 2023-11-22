import pytest
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append("../")

from src.util import *

# # test
# 1) check the output is styler
# 2) check the col number == 2
# 3) check the row number == feature number
# 4) check first column is object
# 5) check second column are number
# 6) check whether the absolute values are sorted descreasingly
# 7) test head
# 7) check errors

def test_fi_output():
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

    output = plot_logistic_regression_feature_importance(test_lr_pipe)
    output_head = plot_logistic_regression_feature_importance(test_lr_pipe, head=test_head)

    assert isinstance(output, pd.io.formats.style.Styler), f"Output should be a Styler (current: {type(output)})."
    assert len(output.columns) == 2, "Output should have 2 columns only."
    #assert output.data.shape[0] == len(test_X.columns), "Number of rows should equal to number of features when `head` is not specified."
    assert output_head.data.shape[0] == test_head, "Number of rows should equal to specified `head`."
    assert output.data.iloc[:,0].dtype == np.dtype('O'), "First column must be a feature string vector."
    assert output.data.iloc[:,1].dtype == np.dtype('float64'), "Second column must be a coefficient numeric vector."
    assert all(output.data.iloc[:,1].abs() == output.data.iloc[:,1].abs().sort_values(ascending=False)), "Output is not sorted by the absolute value of coefficients"
        

def test_fi_error():
    # if len(fitted_lr_pipe.named_steps) != 2:
    #     raise TypeError("`fitted_lr_pipe` is expected to have exactly two components: ColumnTransformer and LogisticRegression")
    
    # if not isinstance(ct, ColumnTransformer):
    #     raise TypeError("1st component in the `fitted_lr_pipe` is expected to be a ColumnTransformer")
    # if not isinstance(lr, LogisticRegression):
    #     raise TypeError("2nd component in the `fitted_lr_pipe` is expected to be a LogisticRegression")
    # if len(ct.named_transformers_) == 0:
    #     raise TypeError("ColumnTransformer has no Encoder")
    # try:
    #     coef = lr.coef_[0]
    # except AttributeError as e:
    #     raise TypeError("LogisticRegression is not fitted (`fitted_lr_pipe.fit(X_train, y_train))")
        
    # if len(features) != len(coef):
    #     raise ValueError("The number of features does not match the number of coefficients")
    pass