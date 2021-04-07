from sklearn import ensemble
from sklearn import pipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingRegressor, StackingClassifier


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
            ("zero_variance_filter", VarianceThreshold(threshold=0)),
            # ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )


def build_ensemble_classification_pipeline():

    models = [
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC()),
        ("RF", RandomForestClassifier()),
        ("bayes", GaussianNB()),
    ]
    final_classifier = LogisticRegression()
    regressor = StackingClassifier(estimators=models, final_estimator=final_classifier)

    return pipeline.Pipeline(
        [
            ("zero_variance_filter", VarianceThreshold(threshold=0)),
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )
