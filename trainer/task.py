import argparse
import os
import numpy as np
import logging
import sys
from trainer import globals
import tensorflow as tf
<<<<<<< HEAD
import subprocess
=======
import random
>>>>>>> 5ab38c9bb32c46cf61e67f81cec628636992cca0


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


def download_datasets():
    dataset_cmd = f"gsutil -m cp -R gs://{globals.dataset.name} ./datasets/"
    oos_cmd = f"gsutil -m cp -R gs://{globals.out_of_study_dataset.name} ./datasets"
    subprocess.run(dataset_cmd.split())
    subprocess.run(oos_cmd.split())


def seed_libraries(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def main():
    """Entry point."""
    random_seed = 69420
    print(random_seed, "random_seed")
    seed_libraries(random_seed)
    flags = _parse_args(sys.argv[1:])
    logging.basicConfig(level=flags.log_level.upper())
    # Set up config and select datasets
    globals.init(in_study=flags.in_study, out_of_study=flags.out_of_study)
    # Trigger the experiment
    if globals.FORCE_GCS_DOWNLOAD:
        download_datasets()
    globals.dataset.run_experiment(flags)


if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    main()
