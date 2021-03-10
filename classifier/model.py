import scipy.stats


from sklearn import ensemble
from sklearn import pipeline
from classifier import globals

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


def print_and_return(data):
    print(data)
    return data


def build_pipeline():

    regressor = ensemble.RandomForestRegressor()

    return pipeline.Pipeline(
        [
            # ("printer", FunctionTransformer(print_and_return)),
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )


def predict_and_evaluate(model, x_test, labels):
    prediction = model.predict(x_test)

    globals.comet_logger.log_confusion_matrix(list(labels), list(prediction))

    rmse = mean_squared_error(prediction, labels, squared=False)
    rmse_per_subject = [
        mean_squared_error([pred], [label], squared=False)
        for pred, label in zip(prediction, labels)
    ]
    return rmse, rmse_per_subject


def evaluate_oos(model, oos_x_test, oos_labels):

    return predict_and_evaluate(model, oos_x_test, oos_labels)


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, labels, oos_x_test, oos_labels):
    (
        rmse,
        rmse_per_subject,
    ) = predict_and_evaluate(model, x_test, labels)

    oos_rmse, oos_rmse_per_subject = evaluate_oos(model, oos_x_test, oos_labels)
    FGI = anosim(rmse_per_subject, oos_rmse_per_subject)
    print("RMSE")
    print(rmse)

    print("OOS RMSE")
    print(oos_rmse)

    print("ANOSIM score - FGI:")
    print(FGI)
    metrics = {
        "rmse": rmse,
        "oos_rmse": oos_rmse,
        "FGI": FGI,
    }
    return metrics


def all_ranks(in_study, out_of_study):
    combined = [*in_study, *out_of_study]
    combined_ranks = scipy.stats.rankdata(combined)
    in_study_ranks = combined_ranks[len(in_study) :]
    out_of_study_ranks = combined_ranks[: len(out_of_study)]

    return in_study_ranks, out_of_study_ranks, combined_ranks


def anosim(in_study, out_of_study):
    in_study_ranks, out_of_study_ranks, combined_ranks = all_ranks(
        in_study, out_of_study
    )
    amount_of_samples = len(combined_ranks)

    return (
        combined_ranks.mean() - (in_study_ranks.mean() - out_of_study_ranks.mean())
    ) / ((amount_of_samples * (amount_of_samples - 1)) / 4)
