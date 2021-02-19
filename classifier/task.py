from comet_ml import Experiment
from classifier import globals
import argparse
import os
import numpy as np
import sys
import tensorflow as tf
import random


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

    return parser.parse_args(argv)


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
        project_name="ideal-pancake",
        workspace="s0lvang",
    )
    experiment.set_name(flags.experiment_name)
    # Set up config and select datasets
    globals.init(
        in_study=flags.in_study,
        out_of_study=flags.out_of_study,
        experiment=experiment,
        _flags=flags,
    )
    # Trigger the experiment
    globals.dataset.run_experiment(flags)


if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    main()
