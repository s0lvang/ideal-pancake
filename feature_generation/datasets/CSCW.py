import pandas as pd
from feature_generation.datasets.Timeseries import Timeseries
from os.path import basename


class CSCW(Timeseries):
    def __init__(self):
        super().__init__("cscw")
        self.column_name_mapping = {
            "id": self.column_names["subject_id"],
            "Fixation Start [ms]": self.column_names["time"],
            "Position X": self.column_names["x"],
            "Position Y": self.column_names["y"],
            "Average Pupil Size [px] X": self.column_names["pupil_diameter"],
            "Fixation Duration [ms]": self.column_names["duration"],
            "Fixation End [ms]": self.column_names["fixation_end"],
        }
        self.label = "Posttest.Score"

    def prepare_files(self, file_references, metadata_references):
        labels = pd.DataFrame()
        dataset = []
        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f, sep=";")
        for file_reference in file_references:
            dataset, labels = self.prepare_file(
                file_reference, metadata_file, dataset, labels
            )
        labels = labels.T
        return dataset, labels

    def prepare_file(self, file_reference, metadata_file, dataset, labels):
        participant_name_array = basename(file_reference.reference).split("_")[0:3]
        participant_name = "_".join(participant_name_array)
        with file_reference.open("r") as f:
            csv = pd.read_csv(f)
        csv = csv.rename(columns=self.column_name_mapping)
        csv[self.column_names["subject_id"]] = participant_name
        dataset.append(csv)
        print(participant_name)
        labels[participant_name] = metadata_file[
            metadata_file["Participant"] == participant_name
        ].iloc[0]
        return dataset, labels

    def __str__(self):
        return super().__str__()
