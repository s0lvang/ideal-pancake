from sklearn import ensemble
from sklearn import pipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor


def print_and_return(data):
    print(data)
    return data


def build_pipeline():

    regressor = ensemble.RandomForestClassifier()

    return pipeline.Pipeline(
        [
            # ("printer", FunctionTransformer(print_and_return)),
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )


def build_ensemble_regression_pipeline():

    models = [
        ("KNN", KNeighborsRegressor()),
        ("SVM", SVR()),
        ("RF", RandomForestRegressor()),
    ]
    final_regressor = LinearRegression()
    regressor = StackingRegressor(estimators=models, final_estimator=final_regressor)

    return pipeline.Pipeline(
        [
            # ("printer", FunctionTransformer(print_and_return)),
            ("zero_variance_filter", VarianceThreshold(threshold=0)),
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )
