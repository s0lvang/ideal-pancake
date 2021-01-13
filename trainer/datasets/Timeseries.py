from trainer.utils import unify_labels
import pandas as pd
from itertools import takewhile
from sklearn import model_selection

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

    def run_experiment(self, flags):
        """Testbed for running model training and evaluation."""
        dataset, labels = self.data_and_labels()
        print("jetris labels")
        print(labels)
        filtered_data = self.get_data_from_feature_selection(dataset).fillna(
            method="ffill"
        )
        (
            indices_train,
            indices_test,
            labels_train,
            labels_test,
            dataset_train,
            dataset_test,
        ) = self.train_test_split(filtered_data, labels)
        pipeline = model.build_pipeline(flags)
        model.set_dataset(pipeline, dataset_train)
        pipeline.fit(indices_train, labels_train)
        oos_data, oos_labels = globals.out_of_study_dataset.data_and_labels()
        unify_labels(oos_labels, labels)
        scores = model.evaluate_model(pipeline, indices_test, labels_test, dataset_test)
        model.store_model_and_metrics(pipeline, scores, flags.job_dir)

    def get_data_from_feature_selection(self, dataset):
        columns_to_use = self.feature_columns + [
            self.column_names["time"],
            self.column_names["subject_id"],
        ]
        return dataset[columns_to_use]

    def train_test_split(self, filtered_data, labels):
        indices = pd.DataFrame(index=labels.index).astype("int64")
        (
            indices_train,
            indices_test,
            labels_train,
            labels_test,
        ) = model_selection.train_test_split(indices, labels)
        dataset_train = filtered_data[
            filtered_data[self.column_names["subject_id"]].isin(indices_train.index)
        ]
        dataset_test = filtered_data[
            filtered_data[self.column_names["subject_id"]].isin(indices_test.index)
        ]
        return (
            indices_train,
            indices_test,
            labels_train,
            labels_test,
            dataset_train,
            dataset_test,
        )

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
