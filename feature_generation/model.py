from sklearn import pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_generation import globals
from feature_generation import utils
from tsfresh.transformers import FeatureAugmenter
from feature_generation.neural_network.vgg16 import extract_features_from_vgg16
from feature_generation.heatmap.generate_heatmaps import create_heatmaps
import pandas as pd


def print_and_return(data):
    print(data)
    return data


def set_dataset(model, dataset):
    model.set_params(augmenter__timeseries_container=dataset)


def build_ts_fresh_extraction_pipeline():
    return pipeline.Pipeline(
        [
            (
                "augmenter",
                FeatureAugmenter(
                    column_id=globals.dataset.column_names["subject_id"],
                    column_sort=globals.dataset.column_names["time"],
                    default_fc_parameters=globals.dataset.tsfresh_features,
                    n_jobs=16,
                ),
            ),
        ]
    )


def create_vgg_pipeline():
    return pipeline.Pipeline(
        [
            ("create_heatmaps", FunctionTransformer(create_heatmaps)),
            ("vgg_16_scaling", FunctionTransformer(utils.preprocess_for_imagenet)),
            ("vgg_16", FunctionTransformer(extract_features_from_vgg16)),
            ("dataframe_from_features", FunctionTransformer(dataframe_from_np_array)),
        ]
    )


def dataframe_from_np_array(x):
    return pd.DataFrame(x, columns=[f"heatmaps_{i}" for i in range(len(x[0]))])
