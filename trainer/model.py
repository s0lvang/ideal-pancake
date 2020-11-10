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


def build_pipeline(flags):
    transform = TSInterpolator(400)

    preprocessor = ColumnConcatenator()

    classifier = TimeSeriesForestClassifier(n_estimators=flags.n_estimators)

    return pipeline.Pipeline(
        [
            ("transform", transform),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
