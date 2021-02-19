from feature_generation.timeseries.tsfresh_custom_calculators import (
    load_custom_functions,
)
import pandas as pd
import numpy as np

from feature_generation.datasets.Dataset import Dataset
from feature_generation import model
from feature_generation import globals


class Timeseries(Dataset):
    def __init__(self, name):
        super().__init__(name)
        self.column_names = {
            "time": "time",
            "subject_id": "subject_id",
            "x": "x",
            "y": "y",
            "pupil_diameter": "pupil_diameter",
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
        indices = get_indicies(labels)
        data = self.select_columns_and_fill_na(data)
        return data, labels, indices

    def select_columns_and_fill_na(self, data):
        df = data[self.columns_to_use]
        data_with_nan = df.replace({0: np.nan})
        return data_with_nan.dropna()

    def generate_features(self):
        data, labels, indices = self.prepare_dataset()

        preprocessing_pipeline = model.build_ts_fresh_extraction_pipeline()
        model.set_dataset(preprocessing_pipeline, data)
        data = preprocessing_pipeline.fit_transform(indices)

        if globals.flags.environment == "remote":
            globals.dataset.upload_features_to_gcs(data, labels)

        print(data)

    def __str__(self):
        return super().__str__()


def get_indicies(labels):
    return pd.DataFrame(index=labels.index).astype("int64")
