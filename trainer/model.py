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

"""ML model definitions."""


import numpy as np
from sklearn import compose
from sklearn import ensemble
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from trainer import metadata
from trainer import utils
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime.classification.dictionary_based import BOSSEnsemble

from sktime.transformers.series_as_features.interpolate import TSInterpolator 


def get_estimator(flags):
    """Generate ML Pipeline which include both pre-processing and model training.

    Args:
      flags: (argparse.ArgumentParser), parameters passed from command-line

    Returns:
      sklearn.pipeline.Pipeline
    """

    classifier = TimeSeriesForestClassifier(
        n_estimators=flags.n_estimators,
    )

    preprocessor = ColumnConcatenator()
    clf = ColumnEnsembleClassifier(
        estimators=[
            ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
            ("BOSSEnsemble3", BOSSEnsemble(max_ensemble_size=5), [3]),
        ]
    )
    estimator = pipeline.Pipeline(
        [
            ("transform", TSInterpolator(400)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return estimator
