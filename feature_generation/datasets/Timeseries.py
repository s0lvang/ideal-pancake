from feature_generation.timeseries.tsfresh_custom_calculators import (
    load_custom_functions,
)
import pandas as pd
import numpy as np
from functools import reduce

from feature_generation.datasets.Dataset import Dataset
from feature_generation import model
from feature_generation import globals
import tsfresh
from feature_generation.eyetracking import saccades, generate_eye_tracking_features
from feature_generation.normalize.normalize import normalize_data


class Timeseries(Dataset):
    def __init__(self, name):
        super().__init__(name)
        self.column_names = {
            "time": "time",
            "subject_id": "subject_id",
            "x": "x",
            "y": "y",
            "pupil_diameter": "pupil_diameter",
            "duration": "duration",
            "fixation_end": "fixation_end",
            "saccade_length": "saccade_length",
            "saccade_angle": "saccade_angle",
            "saccade_duration": "saccade_duration",
        }
        load_custom_functions()
        self.tsfresh_features = {
            "fft_aggregated": [
                {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
            ],
            "lhipa": None,
            "arima": None,
            "garch": None,
            "markov": None,
        }
        self.numeric_features = [
            self.column_names["pupil_diameter"],
        ]
        self.categorical_features = []
        self.feature_columns = self.numeric_features + self.categorical_features
        self.columns_to_use = self.feature_columns + [
            self.column_names["time"],
            self.column_names["subject_id"],
        ]

    def prepare_dataset(self):
        data, labels = self.data_and_labels()
        normalize_data(data)
        generate_eye_tracking_columns(data)
        return data, labels

    def generate_features(self):
        data, labels = self.prepare_dataset()

        heatmap_pipeline = model.create_vgg_pipeline()
        heatmaps_features = heatmap_pipeline.fit_transform(data)
        heatmaps_features.index = labels.index

        eye_tracking_features = (
            generate_eye_tracking_features.generate_eye_tracking_features(data)
        )
        eye_tracking_features.index = labels.index

        data = pd.concat(data)
        time_series_features = tsfresh.extract_features(
            data.loc[:, self.columns_to_use],
            column_id=globals.dataset.column_names["subject_id"],
            column_sort=globals.dataset.column_names["time"],
            default_fc_parameters=globals.dataset.tsfresh_features,
        )
        dataframes = [time_series_features, heatmaps_features, eye_tracking_features]
        data = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True
            ),
            dataframes,
        )
        if globals.flags.environment == "remote":
            globals.dataset.upload_features_to_gcs(data, labels)

        print(data)
        print(labels)

    def __str__(self):
        return super().__str__()


def generate_eye_tracking_columns(data):
    for df in data:
        df["saccade_length"] = saccades.get_saccade_length(df)
        df["saccade_duration"] = saccades.get_saccade_duration(df)
        df["angle_of_saccades"] = saccades.get_angle_of_saccades(df)
    return data


def get_indicies(labels):
    return pd.DataFrame(index=labels.index).astype("int64")
