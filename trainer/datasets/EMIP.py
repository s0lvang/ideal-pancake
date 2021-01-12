import pandas as pd
from itertools import takewhile

from trainer.datasets.Timeseries import Timeseries


class EMIP(Timeseries):
    def __init__(self):
        super().__init__("emip")
        self.label = "expertise_programming"
        self.numeric_features = [
            "Pupil Confidence",
        ]
        self.categorical_features = []
        self.feature_columns = self.numeric_features + self.categorical_features

    def prepare_files(self, file_references, metadata_references):
        print("in prep files emip")
        labels = pd.Series()
        dataset = pd.DataFrame()
        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f)
        for file_reference in file_references:
            with file_reference.open("r") as f:
                dataset, labels = self.prepare_file(f, metadata_file, dataset, labels)
        return dataset, labels

    def prepare_file(self, f, metadata_file, dataset, labels):
        print(f)
        subject_id = get_header(f)["Subject"][0]
        csv = pd.read_csv(f, sep="\t", comment="#")
        csv["id"] = int(subject_id)
        dataset = dataset.append(csv, ignore_index=True)
        labels.at[int(subject_id)] = metadata_file.loc[int(subject_id) - 1, self.label]
        return dataset, labels

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