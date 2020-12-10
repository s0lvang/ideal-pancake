import logging
import os

from sklearn import ensemble
from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from trainer import config
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
                    column_id="id",
                    column_sort="Time",
                    default_fc_parameters=config.TSFRESH_FEATURES,
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
        restore_best_weights=True
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
    if dataset_test:
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
    model_output_path = os.path.join(output_dir, "model", config.MODEL_FILE_NAME)
    metric_output_path = os.path.join(output_dir, "experiment", config.METRIC_FILE_NAME)

    utils.dump_object(model, model_output_path)
    utils.dump_object(metrics, metric_output_path)
