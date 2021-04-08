from classifier.WeightedAverageEnsemble import WeightedAverageEnsemble
from sklearn import ensemble
from sklearn import pipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.decomposition import PCA


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


def build_ensemble_regression_pipeline(dimensionality_reduction_name):

    models = [
        ("KNN", KNeighborsRegressor()),
        ("SVM", SVR()),
        ("RF", RandomForestRegressor()),
    ]
    regressor = WeightedAverageEnsemble(estimators=models)
    if dimensionality_reduction_name == "lasso":
        dimensionality_reduction = SelectFromModel(Lasso(), threshold="median")
    elif dimensionality_reduction_name == "PCA":
        dimensionality_reduction = PCA(n_components=0.75, svd_solver="full")

    return pipeline.Pipeline(
        [
            ("zero_variance_filter", VarianceThreshold(threshold=0)),
            (dimensionality_reduction_name, dimensionality_reduction),
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
