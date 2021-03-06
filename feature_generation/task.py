from comet_ml import Experiment
from feature_generation import globals
from feature_generation.utils import str2bool
import argparse
import os
import numpy as np
import sys
import tensorflow as tf
import subprocess
import random


def _parse_args(argv):
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="""Dataset to use for training and evaluation.
            """,
        required=True,
    )

    parser.add_argument(
        "--job-dir",
        help="Output directory for exporting model and other metadata.",
        required=True,
    )

    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        required=True,
    )

    parser.add_argument(
        "--environment",
        help="local or remote",
        default="local",
        type=str,
    )

    parser.add_argument(
        "--comet_api_key",
        help="apikey",
        default=1,
        type=str,
    )

    parser.add_argument(
        "--download_files",
        help="Should files be downloaded",
        default=False,
        type=str2bool,
        nargs="?",
        const="",
    )

    return parser.parse_args(argv)


def download_datasets():
    dataset_cmd = f"gsutil -m cp -R gs://{globals.dataset.name} ./datasets/"
    subprocess.run(dataset_cmd.split())


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
    experiment = Experiment(
        api_key=flags.comet_api_key,
        project_name="ideal-pancake-feature-generation",
        workspace="s0lvang",
    )
    experiment.set_name(flags.experiment_name)
    # Set up config and select datasets
    globals.init(
        dataset_name=flags.dataset,
        experiment=experiment,
        _flags=flags,
    )
    # Trigger the experiment
    if flags.download_files or flags.environment == "remote":
        download_datasets()
    globals.dataset.generate_features()


if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    main()
