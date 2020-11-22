import logging
import os

from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import metadata
from trainer import utils
from trainer.cnnlstm.lstm import create_model
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import MinimalFCParameters
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import model_selection
from sklearn.metrics import classification_report


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
                    default_fc_parameters=metadata.TSFRESH_FEATURES,
                ),
            ),
            ("printer", FunctionTransformer(print_and_return, print_and_return)),
            ("classifier", classifier),
        ]
    )


def build_lstm_pipeline(shape, classes):
    model = create_model(classes=classes, *shape)
    classifier = KerasClassifier(model, epochs=100, batch_size=500, verbose=0)
    return pipeline.Pipeline(
        [
            ("classifier", classifier),
        ]
    )


# This method handles all evaluation of the model. Since we don't actually need the prediction for anything it is also handled in here.
def evaluate_model(model, x_test, y_test, dataset_test=None):
    if dataset_test:
        set_dataset(model, dataset_test)
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
