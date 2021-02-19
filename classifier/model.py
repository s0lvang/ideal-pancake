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
            # ("Lasso", SelectFromModel(Lasso())),
            ("classifier", regressor),
        ]
    )


def build_lasso_pipeline():
    classifier = RandomForestRegressor()
    return pipeline.Pipeline(
        [
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", classifier),
        ],
    )


def predict_and_evaluate(model, x_test, labels):
    prediction = model.predict(x_test)
    prediction = labels.get_clusters_from_values(prediction)
    y_test = labels.get_clusters_from_values(labels.test)

    denormalized_prediction = labels.denormalize_labels(prediction)
    globals.comet_logger.log_confusion_matrix(
        list(labels.original_labels_test), list(denormalized_prediction)
    )

    scaling_factor = labels.original_max - labels.original_min
    nrmses = nrmse_per_subject(
        predicted_values=prediction,
        original_values=y_test,
        scaling_factor=scaling_factor,
    )
    rmse = mean_squared_error(prediction, y_test, squared=False)
    nrmse = normalized_root_mean_squared_error(prediction, y_test, scaling_factor)
    return nrmses, rmse, nrmse


def evaluate_oos(model, oos_x_test, oos_labels):

    return predict_and_evaluate(model, oos_x_test, oos_labels)


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, labels, oos_x_test, oos_labels):
    (
        nrmses,
        rmse,
        nrmse,
    ) = predict_and_evaluate(model, x_test, labels)

    oos_nrmses, oos_rmse, oos_nrmse = evaluate_oos(model, oos_x_test, oos_labels)
    FGI = anosim(nrmses, oos_nrmses)
    print("RMSE")
    print(rmse)

    print("NRMSE")
    print(nrmse)

    print("OOS RMSE")
    print(oos_rmse)

    print("OOS NRMSE")
    print(oos_nrmse)

    print("ANOSIM score - FGI:")
    print(FGI)
    metrics = {
        "rmse": rmse,
        "nrmse": nrmse,
        "oos_rmse": oos_rmse,
        "oos_nrmse": oos_nrmse,
        "FGI": FGI,
    }
    globals.comet_logger.log_metrics(metrics)


def nrmse_per_subject(predicted_values, original_values, scaling_factor):
    if scaling_factor == 0:
        raise ZeroDivisionError(
            "The observations in the ground truth are constant, we would get a divide by zero error."
        )
    return [
        normalized_root_mean_squared_error(
            [predicted_value], [original_value], scaling_factor
        )
        for predicted_value, original_value in zip(predicted_values, original_values)
    ]


def normalized_root_mean_squared_error(
    predicted_value,
    original_value,
    scaling_factor,
):
    return (
        100
        * mean_squared_error(predicted_value, original_value, squared=False)
        / scaling_factor
    )


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


def get_label_from_range(value, ranges):
    for key, range in ranges.items():
        if value > range[0] and value < range[1]:
            return key
    raise Exception("not in range")
