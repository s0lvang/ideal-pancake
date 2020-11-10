import logging
import os

import numpy as np

from sklearn import compose
from sklearn import ensemble
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report

from trainer import metadata
from trainer import utils

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.transformers.series_as_features.interpolate import TSInterpolator


def build_pipeline(flags):
    transform = TSInterpolator(400)

    preprocessor = ColumnConcatenator()

    classifier = TimeSeriesForestClassifier(n_estimators=flags.n_estimators)

    return pipeline.Pipeline(
        [
            ("transform", transform),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    print(classification_report(y_test, prediction))
    # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)
    scores = model_selection.cross_val_score(model, x_test, y_test, cv=2)
    logging.info(scores)

    return scores


# Write model and eval metrics to `output_dir`
def store_model_and_metrics(model, metrics, output_dir):
    model_output_path = os.path.join(output_dir, "model", metadata.MODEL_FILE_NAME)
    metric_output_path = os.path.join(
        output_dir, "experiment", metadata.METRIC_FILE_NAME
    )

    utils.dump_object(model, model_output_path)
    utils.dump_object(metrics, metric_output_path)
