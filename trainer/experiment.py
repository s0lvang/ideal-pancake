import hypertune
import numpy as np
from sklearn import model_selection

from trainer import metadata
from trainer import model
from trainer import utils


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    dataset, labels = utils.read_emip_from_gcs()
    filtered_data = get_data_from_feature_selection(dataset)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        filtered_data, labels
    )
    pipeline = model.build_pipeline(flags)
    pipeline.fit(x_train, y_train)
    scores = model.evaluate_model(pipeline, x_test, y_test)
    model.store_model_and_metrics(pipeline, scores, flags.job_dir)

    # Tuning hyperparameters, currently unused
    # hypertune(scores)


def get_data_from_feature_selection(dataset):
    columns_to_use = metadata.FEATURE_COLUMNS
    return dataset[columns_to_use]


def hypertune(metrics):
    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name
    # The only functional difference is that if you use a custom name,
    # you must set the hyperparameterMetricTag value in the
    # HyperparameterSpec object in your job request to match your chosen name.
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="my_metric_tag",
        metric_value=np.mean(metrics),
        global_step=1000,
    )
