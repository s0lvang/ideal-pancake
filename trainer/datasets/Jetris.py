import pandas as pd

from trainer import experiment
from trainer.datasets.Dataset import Dataset


class Jetris(Dataset):
    def __init__(self):
        super().__init__("jetris")
        self.tsfresh_features = {
            "length": None,
            "fft_aggregated": [
                {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
            ],
            "fft_coefficient": [{"coeff": k, "attr": "real"} for k in range(100)],
        }
        self.numeric_features = [
            "Speed",
        ]
        self.categorical_features = []
        self.feature_columns = self.numeric_features + self.categorical_features
        self.experimenter = experiment.run_ts_experiment

    def prepare_files(self, file_references, metadata_references):
        print("in prep files jetris")
        labels = pd.Series()
        dataset = pd.DataFrame()
        for file_reference in file_references:
            with file_reference.open("r") as f:
                dataset, labels = self.prepare_file(f, dataset, labels)
        # labels = convert_labels_to_categorical()
        dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
        return dataset, labels

    def prepare_file(self, f, dataset, labels):
        print(f)
        csv = pd.read_csv(f, comment="#")
        csv = csv[
            csv["Pupil.initial"] != "saccade"
        ]  # this drops all lines that are saccades, we should do something smarter here.
        game_id = csv["gameID"][0]
        dataset = dataset.append(csv, ignore_index=True)
        labels.at[int(game_id)] = csv["Score.1"].iloc[-1]
        return dataset, labels

    def __str__(self):
        return super().__str__()