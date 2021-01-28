import os
import math
import scipy.stats


from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import globals
from trainer import utils
from trainer.neural_network.vgg16 import (
    create_model_factory,
    extract_features_from_vgg16,
)
from trainer.neural_network.TensorboardCallback import BucketTensorBoard
from tsfresh.transformers import FeatureAugmenter
from scikeras.wrappers import KerasRegressor

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso


def print_and_return(data):
    print(data)
    return data


def set_dataset(model, dataset):
    model.set_params(augmenter__timeseries_container=dataset)


def build_pipeline(flags):

    regressor = ensemble.RandomForestRegressor(n_estimators=flags.n_estimators)

    return pipeline.Pipeline(
        [
            (
                "augmenter",
                FeatureAugmenter(
                    column_id=globals.dataset.column_names["subject_id"],
                    column_sort=globals.dataset.column_names["time"],
                    default_fc_parameters=globals.dataset.tsfresh_features,
                ),
            ),
            ("printer", FunctionTransformer(print_and_return)),
            ("regressor", regressor),
        ]
    )


def build_lasso_pipeline():
    classifier = RandomForestRegressor()
    return pipeline.Pipeline(
        [
            ("vgg_16_scaling", FunctionTransformer(utils.preprocess_for_imagenet)),
            ("vgg_16", FunctionTransformer(extract_features_from_vgg16)),
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", classifier),
        ]
    )


def build_lstm_pipeline(shape, classes, output_dir):
    model_factory = create_model_factory(classes=classes, *shape)
    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=50,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )
    tensorboard_callback = BucketTensorBoard(output_dir, histogram_freq=1)
    preprocessing = FunctionTransformer(
        utils.preprocess_for_imagenet, check_inverse=False
    )
    classifier = KerasRegressor(
        build_fn=model_factory,
        epochs=1,
        batch_size=1,
        verbose=2,
        fit__validation_split=0.2,
        callbacks=[tensorboard_callback, earlystopping_callback],
    )
    return pipeline.Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )


def predict_and_evaluate(model, x_test, y_test, ranges):
    prediction = model.predict(x_test)
    prediction = [get_label_from_range(x, ranges) for x in prediction]
    y_test = [get_label_from_range(x, ranges) for x in y_test]
    scaling_factor = max(y_test) - min(y_test)
    nrmses = nrmse_per_subject(
        predicted_values=prediction,
        original_values=y_test,
        scaling_factor=scaling_factor,
    )
    rmse = mean_squared_error(prediction, y_test, squared=False)
    nrmse = normalized_root_mean_squared_error(prediction, y_test, scaling_factor)
    return nrmses, rmse, nrmse


def evaluate_oos(model, oos_x_test, oos_y_test, oos_ranges, oos_dataset):
    if oos_dataset is not None:
        set_dataset(model, oos_dataset)

    return predict_and_evaluate(model, oos_x_test, oos_y_test, oos_ranges)


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(
    model, x_test, y_test, oos_x_test, oos_y_test, ranges, oos_ranges, oos_dataset=None
):
    (
        nrmses,
        rmse,
        nrmse,
    ) = predict_and_evaluate(model, x_test, y_test, ranges)

    oos_nrmses, oos_rmse, oos_nrmse = evaluate_oos(
        model, oos_x_test, oos_y_test, oos_ranges, oos_dataset
    )
    print("RMSE")
    print(rmse)

    print("NRMSE")
    print(nrmse)

    print("OOS RMSE")
    print(oos_rmse)

    print("OOS NRMSE")
    print(oos_nrmse)

    print("ANOSIM score - FGI:")
    print(anosim(nrmses, oos_nrmses))


# Write model and eval metrics to `output_dir`
def store_model_and_metrics(model, metrics, output_dir):
    model_output_path = os.path.join(output_dir, "model", globals.MODEL_FILE_NAME)
    metric_output_path = os.path.join(
        output_dir, "experiment", globals.METRIC_FILE_NAME
    )

    utils.dump_object(model, model_output_path)
    utils.dump_object(metrics, metric_output_path)


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
    for key, ranges in ranges.items():
        if value > ranges[0] and value < ranges[1]:
            return key
    raise Exception("not in range")
