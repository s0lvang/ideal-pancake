from feature_generation.normalize.rolling_mean import rolling_mean
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
from feature_generation.eyetracking.saccades import generate_saccade_columns
from feature_generation.normalize.normalize import normalize_data
from feature_generation.eyetracking import generate_eye_tracking_features


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
            "arma": 4,
            "garch": 4,
            "markov": 8,
        }
        self.timeseries_columns = [
            f"{self.column_names['pupil_diameter']}_rolling",
            f"{self.column_names['duration']}_rolling",
            f"{self.column_names['saccade_length']}_rolling",
            f"{self.column_names['saccade_duration']}_rolling",
        ]
        self.tsfresh_columns = self.timeseries_columns + [
            self.column_names["time"],
            self.column_names["subject_id"],
        ]

    def prepare_dataset(self):
        data, labels = self.data_and_labels()
        data = normalize_data(data)
        data = generate_saccade_columns(data)
        data = rolling_mean(data)
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
            data.loc[:, self.tsfresh_columns],
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


def get_indicies(labels):
    return pd.DataFrame(index=labels.index).astype("int64")
