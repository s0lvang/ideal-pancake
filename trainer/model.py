import os
import math
import scipy.stats
from tempfile import mkdtemp


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
from tsfresh.feature_extraction import ComprehensiveFCParameters

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from joblib import Memory
from keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor


def print_and_return(data):
    print(data)
    return data


def set_dataset(model, dataset):
    model.set_params(augmenter__timeseries_container=dataset)


def build_pipeline():

    regressor = ensemble.RandomForestRegressor()

    return pipeline.Pipeline(
        [
            (
                "augmenter",
                FeatureAugmenter(
                    column_id=globals.dataset.column_names["subject_id"],
                    column_sort=globals.dataset.column_names["time"],
                    default_fc_parameters=globals.dataset.tsfresh_features,
                    n_jobs=16,
                ),
            ),
            ("printer", FunctionTransformer(print_and_return)),
            ("Lasso", SelectFromModel(Lasso())),
            ("regressor", regressor),
        ]
    )


def create_vgg_pipeline():
    return pipeline.Pipeline(
        [
            ("vgg_16_scaling", FunctionTransformer(utils.preprocess_for_imagenet)),
            ("vgg_16", FunctionTransformer(extract_features_from_vgg16)),
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


def predict_and_evaluate(model, x_test, labels):
    prediction = model.predict(x_test)
    prediction = labels.get_clusters_from_values(prediction)
    scaling_factor = labels.original_max - labels.original_min
    nrmses = nrmse_per_subject(
        predicted_values=prediction,
        original_values=labels.original_labels_test,
        scaling_factor=scaling_factor,
    )
    rmse = mean_squared_error(prediction, labels.original_labels_test, squared=False)
    nrmse = normalized_root_mean_squared_error(
        prediction, labels.original_labels_test, scaling_factor
    )
    return nrmses, rmse, nrmse


def evaluate_oos(model, oos_x_test, oos_labels, oos_dataset):
    if oos_dataset is not None:
        set_dataset(model, oos_dataset)

    return predict_and_evaluate(model, oos_x_test, oos_labels)


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, labels, oos_x_test, oos_labels, oos_dataset=None):
    (
        nrmses,
        rmse,
        nrmse,
    ) = predict_and_evaluate(model, x_test, labels)

    oos_nrmses, oos_rmse, oos_nrmse = evaluate_oos(
        model, oos_x_test, oos_labels, oos_dataset
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
    print(in_study, "in_study")
    print(out_of_study, "out_of_study")
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
