from trainer.utils import normalize_and_numericalize
import pandas as pd
from itertools import takewhile
from sklearn import model_selection
import numpy as np

from trainer.datasets.Dataset import Dataset
from trainer import model
from trainer import globals


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
        self.tsfresh_features = {
            "length": None,
            "fft_aggregated": [
                {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
            ],
            "fft_coefficient": [{"coeff": k, "attr": "real"} for k in range(100)],
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

    def prepare_dataset(self, data, labels):
        indices = get_indicies(labels)
        data = self.select_columns_and_fill_na(data)
        labels, ranges = normalize_and_numericalize(labels)
        return data, labels, indices, ranges

    def prepare_datasets(self):
        data, labels = self.data_and_labels()
        data, labels, indices, ranges = self.prepare_dataset(data, labels)

        oos_data, oos_labels = globals.out_of_study_dataset.data_and_labels()
        oos_data, oos_labels, oos_indices, oos_ranges = self.prepare_dataset(
            oos_data, oos_labels
        )

        return (
            data,
            labels,
            indices,
            ranges,
            oos_data,
            oos_labels,
            oos_indices,
            oos_ranges,
        )

    def run_experiment(self, flags):
        """Testbed for running model training and evaluation."""
        (
            data,
            labels,
            indices,
            ranges,
            oos_data,
            oos_labels,
            oos_indices,
            oos_ranges,
        ) = self.prepare_datasets()

        (
            indices_train,
            indices_test,
            labels_train,
            labels_test,
        ) = model_selection.train_test_split(indices, labels)

        pipeline = model.build_pipeline()
        model.set_dataset(pipeline, data)
        pipeline.fit(indices_train, labels_train)

        scores = model.evaluate_model(
            model=pipeline,
            x_test=indices_test,
            y_test=labels_test,
            ranges=ranges,
            oos_x_test=oos_indices,
            oos_y_test=oos_labels,
            oos_ranges=oos_ranges,
            oos_dataset=oos_data,
        )

        model.store_model_and_metrics(pipeline, scores, flags.job_dir)

    def select_columns_and_fill_na(self, data):
        df = data[self.columns_to_use]
        data_with_nan = df.replace({0: np.nan})
        return data_with_nan.dropna()

    def __str__(self):
        return super().__str__()


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header


def get_indicies(labels):
    return pd.DataFrame(index=labels.index).astype("int64")
