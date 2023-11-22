import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

def plot_logistic_regression_feature_importance(fitted_lr_pipe, head=None, precision=3, cmap="PiYG", vmin=None, vmax=None, caption=None):
    """
    Returns two EDA plots, one for numerical columns and another for categorical columns.

    Parameters:
    ----------
    data : pd.DataFrame
        input data that is used for plotting
    numerical_cols : list
        list of numerical columns that are to be plotted
    categorical_cols : list
        list of categorical columns that are to be plotted

    Returns:
    ----------
    (alt.Chart, alt.Chart)
        two EDA plots, one for numerical columns and another for categorical columns
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

"""
# handling (assumed fitted estimator, i.e. have only CT and model)

1) len(named_steps) must be == 2
2) 1st must be column transformer 
2.1) the column transformer len must be > 0
3) 2nd must be logistic regression model
4) check coef_ exist in lr_pipe 
5) coef_ len = feature len

# test
1) check the output is styler
2) check the col number == 2
3) check the row number == feature number
4) check first column is object
5) check second column are number
6) check whether the absolute values are sorted descreasingly
7) check errors
"""