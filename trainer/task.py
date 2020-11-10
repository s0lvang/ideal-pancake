# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Executes model training and evaluation."""

import argparse
import logging
import os
import sys

from trainer import experiment


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
    experiment.run_experiment(flags)


if __name__ == "__main__":
    main()
