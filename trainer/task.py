import argparse
import logging
import os
import sys

import hypertune
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
import pandas as pd
from trainer import metadata
from trainer import model
from trainer import utils


def _train_and_evaluate(estimator, dataset, labels, output_dir):

    dataset_with_columns_to_use = utils.filter_columns(dataset).fillna(method="ffill")

    X = pd.DataFrame(index=labels.index).astype("int64")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, labels)

    df_ts_train = dataset_with_columns_to_use[
        dataset_with_columns_to_use["id"].isin(y_train.index)
    ]
    df_ts_test = dataset_with_columns_to_use[
        dataset_with_columns_to_use["id"].isin(y_test.index)
    ]

    estimator.set_params(augmenter__timeseries_container=df_ts_train)
    estimator.fit(x_train, y_train)

    estimator.set_params(augmenter__timeseries_container=df_ts_test)
    prediction = estimator.predict(x_test)
    # Note: for now, use `cross_val_score` defaults (i.e. 3-fold)
    scores = model_selection.cross_val_score(estimator, x_test, y_test, cv=2)

    logging.info(scores)
    print(classification_report(y_test, prediction))
    # Write model and eval metrics to `output_dir`
    model_output_path = os.path.join(output_dir, "model", metadata.MODEL_FILE_NAME)

    metric_output_path = os.path.join(
        output_dir, "experiment", metadata.METRIC_FILE_NAME
    )

    utils.dump_object(estimator, model_output_path)
    utils.dump_object(scores, metric_output_path)

    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name
    # The only functional difference is that if you use a custom name,
    # you must set the hyperparameterMetricTag value in the
    # HyperparameterSpec object in your job request to match your chosen name.
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="my_metric_tag",
        metric_value=np.mean(scores),
        global_step=1000,
    )


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    # Get data for training and evaluation
    dataset, labels = utils.read_emip_from_gcs()
    # Get model
    estimator = model.get_estimator(flags)

    # Run training and evaluation
    _train_and_evaluate(estimator, dataset, labels, flags.job_dir)


def _parse_args(argv):
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="""Dataset to use for training and evaluation.
              Can be BigQuery table or a file (CSV).
              If BigQuery table, specify as as PROJECT_ID.DATASET.TABLE_NAME.
            """,
        required=True,
    )

    parser.add_argument(
        "--job-dir",
        help="Output directory for exporting model and other metadata.",
        required=True,
    )

    parser.add_argument(
        "--log_level",
        help="Logging level.",
        choices=[
            "DEBUG",
            "ERROR",
            "FATAL",
            "INFO",
            "WARN",
        ],
        default="INFO",
    )

    parser.add_argument(
        "--num_samples",
        help="Number of samples to read from `input`",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n_estimators",
        help="Number of trees in the forest.",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--max_depth",
        help="The maximum depth of the tree.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min_samples_leaf",
        help="The minimum number of samples required to be at a leaf node.",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--criterion",
        help="The function to measure the quality of a split.",
        choices=[
            "gini",
            "entropy",
        ],
        default="gini",
    )

    return parser.parse_args(argv)


def main():
    """Entry point."""

    flags = _parse_args(sys.argv[1:])
    logging.basicConfig(level=flags.log_level.upper())
    run_experiment(flags)


if __name__ == "__main__":
    main()
