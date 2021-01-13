import os
import math
import scipy.stats


from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import globals
from trainer import utils
from trainer.cnnlstm.lstm import create_model_factory, root_mean_squared_error
from trainer.cnnlstm.TensorboardCallback import BucketTensorBoard
from tsfresh.transformers import FeatureAugmenter
from scikeras.wrappers import KerasRegressor


from sklearn import model_selection
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping


def print_and_return(data):
    print(data)
    return data


def set_dataset(model, dataset):
    model.set_params(augmenter__timeseries_container=dataset)


def build_pipeline(flags):

    classifier = ensemble.RandomForestClassifier(n_estimators=flags.n_estimators)

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
        epochs=300,
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


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, y_test, dataset_test=None):
    if dataset_test is not None:
        set_dataset(model, dataset_test)
    # print(x_test[0])
    # print(x_test.shape)
    prediction = model.predict(x_test)
    print("Prediction - y test - nrmse_per_subject")
    print(prediction)
    print(y_test)
    print(nrmse_per_subject(predicted_values=prediction, original_values=y_test))

    # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)


# Write model and eval metrics to `output_dir`
def store_model_and_metrics(model, metrics, output_dir):
    model_output_path = os.path.join(output_dir, "model", globals.MODEL_FILE_NAME)
    metric_output_path = os.path.join(
        output_dir, "experiment", globals.METRIC_FILE_NAME
    )

    utils.dump_object(model, model_output_path)
    utils.dump_object(metrics, metric_output_path)


def nrmse_per_subject(predicted_values, original_values):
    scaling_factor = max(original_values) - min(original_values)
    return [
        nrmse(predicted_value, original_value, scaling_factor)
        for predicted_value, original_value in zip(predicted_values, original_values)
    ]


def rmse(predicted_value, original_value):
    return math.sqrt((predicted_value - original_value) ** 2)


def nrmse(
    predicted_value,
    original_value,
    scaling_factor,
):
    return 100 * rmse(predicted_value, original_value) / scaling_factor


def all_ranks(in_study, out_of_study):
    combined = [*in_study, *out_of_study]
    combined_ranks = scipy.stats.rankdata(combined)
    in_study_ranks = combined_ranks[: len(in_study)]
    out_of_study_ranks = combined_ranks[len(in_study) :]

    return in_study_ranks, out_of_study_ranks, combined_ranks


def anosim(in_study, out_of_study):
    in_study_ranks, out_of_study_ranks, combined_ranks = all_ranks(
        in_study, out_of_study
    )
    amount_of_samples = len(combined_ranks)

    return (in_study_ranks.mean() - combined_ranks.mean()) / (
        (amount_of_samples * (amount_of_samples - 1)) / 4
    )
