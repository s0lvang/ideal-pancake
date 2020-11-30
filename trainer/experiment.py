import hypertune
import numpy as np
from sklearn import model_selection

from trainer import metadata
from trainer import model
from trainer import utils
from trainer.datasets import datasets
import pandas as pd


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    dataset, labels = datasets.datasets_and_labels()
    filtered_data = get_data_from_feature_selection(dataset).fillna(method="ffill")
    (
        indices_train,
        indices_test,
        labels_train,
        labels_test,
        dataset_train,
        dataset_test,
    ) = train_test_split(filtered_data, labels)
    pipeline = model.build_pipeline(flags)
    model.set_dataset(pipeline, dataset_train)
    pipeline.fit(indices_train, labels_train)

    scores = model.evaluate_model(pipeline, indices_test, labels_test, dataset_test)
    model.store_model_and_metrics(pipeline, scores, flags.job_dir)


def get_data_from_feature_selection(dataset):
    columns_to_use = metadata.FEATURE_COLUMNS + ["Time", "id"]
    return dataset[columns_to_use]


def train_test_split(filtered_data, labels):
    indices = pd.DataFrame(index=labels.index).astype("int64")
    (
        indices_train,
        indices_test,
        labels_train,
        labels_test,
    ) = model_selection.train_test_split(indices, labels)
    dataset_train = filtered_data[filtered_data["id"].isin(indices_train.index)]
    dataset_test = filtered_data[filtered_data["id"].isin(indices_test.index)]
    return (
        indices_train,
        indices_test,
        labels_train,
        labels_test,
        dataset_train,
        dataset_test,
    )


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
