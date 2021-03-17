# from feature_generation.utils import convert_categorical_labels_to_numerical
from feature_generation.Labels import Labels
import pandas as pd
from itertools import takewhile
import time
from feature_generation.datasets.Timeseries import Timeseries


class EMIP(Timeseries):
    def __init__(self):
        super().__init__("emip-fixations")
        self.column_name_mapping = {
            "id": self.column_names["subject_id"],
            "fixationStart": self.column_names["time"],
            "x": self.column_names["x"],
            "y": self.column_names["y"],
            "averagePupilSize": self.column_names["pupil_diameter"],
            "duration": self.column_names["duration"],
            "fixationEnd": self.column_names["fixation_end"],
        }
        self.label = "expertise_programming"

    def prepare_files(self, file_references, metadata_references):
        labels = pd.DataFrame()
        dataset = []
        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f)
        for file_reference in file_references:
            with file_reference.open("r") as f:
                dataset, labels = self.prepare_file(f, metadata_file, dataset, labels)
        labels = labels.T
        print(labels)
        return dataset, labels

    def prepare_file(self, f, metadata_file, dataset, labels):
        subject_id = int(get_header(f)["Subject"][0])
        csv = pd.read_csv(f, sep="\t", comment="#", engine="c")
        csv = csv.rename(columns=self.column_name_mapping)
        csv[self.column_names["subject_id"]] = subject_id
        csv = csv[csv["status"] == "READING"]
        csv = self.normalize_trials(csv)
        dataset.append(csv)
        labels[subject_id] = metadata_file[
            metadata_file["id"].astype(int) == subject_id
        ].iloc[0]
        return dataset, labels

    def normalize_trials(self, df):
        # Since we combine the trials and only look at sections where they are reading code, it is a huge gap in time in the middle
        trial_1_endtime = df.loc[df["trial_number"] == 1, "fixation_end"].iloc[-1]
        trial_2_starttime = df.loc[df["trial_number"] == 2, "time"].iloc[0]
        diff = trial_2_starttime - trial_1_endtime
        df.loc[df["trial_number"] == 2, "time"] -= diff
        df.loc[df["trial_number"] == 2, "fixation_end"] -= diff
        return df

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
