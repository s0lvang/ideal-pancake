import argparse
import logging
import os
import sys
from trainer import experiment
from trainer import globals
import tensorflow as tf


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
    if flags.input == "emip-images":
        globals.init_emip_images()
        experiment.run_heatmap_experiment(flags)
    elif flags.input == "mooc-images":
        globals.init_mooc_images()
        experiment.run_heatmap_experiment(flags)
    elif flags.input == "emip":
        globals.init_emip()
        experiment.run_ts_experiment(flags)
    elif flags.input == "jetris":
        globals.init_jetris()
        experiment.run_ts_experiment(flags)
    else:
        raise ValueError(f"{flags.input} is not a valid dataset name.")


if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    main()
