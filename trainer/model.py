import numpy as np
import pandas as pd
from sklearn import compose
from sklearn import ensemble
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from trainer import metadata
from trainer import utils
from numpy.fft import fft, ifft
from tsfresh import feature_extraction
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import MinimalFCParameters

from sklearn.base import BaseEstimator, TransformerMixin

def column_fft(data):
    for index, row in data.iterrows():
        for key in data.keys():
            data.at[index, key] = fft(data.at[index, key], 30).astype("float64")
    return data


def inverse_fft(data):
    for index, row in data.iterrows():
        for key in data.keys():
            data.at[index, key] = ifft(data.at[index, key], 30).astype("float64")
    return data


"""
def column_fft(data):
    return data.applymap(lambda x: fft(np.array(x, dtype=float)))

def inverse_fft(data):
    return data.applymap(lambda x: ifft(np.array(x, dtype=float)))
"""
def print_and_return(data):
    print(data)
    return data
 


def get_estimator(flags):

    """Generate ML Pipeline which include both pre-processing and model training.

    Args:
      flags: (argparse.ArgumentParser), parameters passed from command-line

    Returns:
      sklearn.pipeline.Pipeline
    """

    classifier = ensemble.RandomForestClassifier(
    )
    fft_transformer = FunctionTransformer(column_fft, inverse_fft, check_inverse=False)
    
    estimator = pipeline.Pipeline(
        [
            ('augmenter', FeatureAugmenter(column_id='id', column_sort='Time')),
            ('printer', FunctionTransformer(print_and_return, print_and_return)),
            ("classifier", classifier),
        ]
    )

    return estimator

