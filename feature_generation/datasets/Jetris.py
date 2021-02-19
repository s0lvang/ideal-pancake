import pandas as pd

from feature_generation.datasets.Timeseries import Timeseries


class Jetris(Timeseries):
    def __init__(self):
        super().__init__("jetris")
        self.column_name_mapping = {
            "gameID": self.column_names["subject_id"],
            "time[milliseconds]": self.column_names["time"],
            "Pupil.size": self.column_names["pupil_diameter"],
        }

    def prepare_files(self, file_references, metadata_references):
        labels = pd.Series()
        dataset = pd.DataFrame()
        for file_reference in file_references:
            with file_reference.open("r") as f:
                dataset, labels = self.prepare_file(f, dataset, labels)
        return dataset, labels

    def prepare_file(self, f, dataset, labels):
        csv = pd.read_csv(f, comment="#")
        csv = csv[
            csv["Pupil.initial"] != "saccade"
        ]  # this drops all lines that are saccades, we should do something smarter here.
        csv = csv.rename(columns=self.column_name_mapping)
        game_id = csv[self.column_names["subject_id"]][0]
        if csv["Score.1"].iloc[-1] == 0:
            return dataset, labels
        dataset = dataset.append(csv, ignore_index=True)
        labels.at[int(game_id)] = csv["Score.1"].iloc[-1]
        return dataset, labels

    def __str__(self):
        return super().__str__()