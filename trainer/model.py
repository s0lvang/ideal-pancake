import logging
import os

from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import globals
from trainer import utils
from trainer.cnnlstm.lstm import (
    create_model_factory,
    extract_features,
    root_mean_squared_error,
)
from trainer.cnnlstm.TensorboardCallback import BucketTensorBoard
from tsfresh.transformers import FeatureAugmenter
from scikeras.wrappers import KerasRegressor

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.svm import LinearSVC


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


def build_lstm_pipeline():
    classifier = RandomForestRegressor()
    return pipeline.Pipeline(
        [
            ("Lasso", SelectFromModel(Lasso())),
            ("classifier", classifier),
        ]
    )


def extract_features_vgg16(X):
    subjects = utils.preprocess_for_imagenet(X)
    return extract_features(subjects)


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, y_test, dataset_test=None):
    if dataset_test is not None:
        set_dataset(model, dataset_test)
    print(x_test[0])
    print(x_test.shape)
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
