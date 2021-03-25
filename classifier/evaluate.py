import pandas as pd
from classifier import globals
from sklearn.metrics import mean_squared_error
import scipy.stats


def predict_and_evaluate(model, x_test, labels):
    prediction = model.predict(x_test)
    df_predictions = pd.DataFrame()
    df_predictions["prediction"] = prediction
    df_predictions["true"] = list(labels)
    print(labels)
    print(df_predictions)
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
