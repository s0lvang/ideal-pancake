import argparse
import logging
import sys
from trainer import globals
import tensorflow as tf


def _parse_args(argv):
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_study",
        help="""Dataset to use for training and evaluation.
            """,
        required=True,
    )

    parser.add_argument(
        "--out_of_study",
        help="""Dataset to use for evaluating FGI.
            """,
        required=False,
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
    # Set up config and select datasets
    globals.init(in_study=flags.in_study, out_of_study=flags.out_of_study)
    # Trigger the experiment
    globals.dataset.run_experiment(flags)


if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    main()
