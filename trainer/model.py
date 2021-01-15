import logging
import os

from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import globals
from trainer import utils
from trainer.neural_network.vgg16 import (
    create_model_factory,
    extract_features_from_vgg16,
    root_mean_squared_error
)
from trainer.neural_network.TensorboardCallback import BucketTensorBoard
from tsfresh.transformers import FeatureAugmenter
from scikeras.wrappers import KerasRegressor

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

    classifier = ensemble.RandomForestClassifier(n_estimators=flags.n_estimators)

    return pipeline.Pipeline(
        [
            (
                "augmenter",
                FeatureAugmenter(
                    column_id="id",
                    column_sort="Time",
                    default_fc_parameters=globals.config.TSFRESH_FEATURES,
                ),
            ),
            ("printer", FunctionTransformer(print_and_return)),
            ("classifier", classifier),
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
    prediction = model.predict(x_test)
    print(prediction)
    print(y_test)
    print(root_mean_squared_error(y_test, prediction))
    # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)


# Write model and eval metrics to `output_dir`
def store_model_and_metrics(model, metrics, output_dir):
    model_output_path = os.path.join(
        output_dir, "model", globals.config.MODEL_FILE_NAME
    )
    metric_output_path = os.path.join(
        output_dir, "experiment", globals.config.METRIC_FILE_NAME
    )

    utils.dump_object(model, model_output_path)
    utils.dump_object(metrics, metric_output_path)
