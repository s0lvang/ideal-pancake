import pandas as pd
from feature_generation.datasets.Timeseries import Timeseries


class Fractions(Timeseries):
    def __init__(self):
        super().__init__("fractions")
        self.column_name_mapping = {
            "id": self.column_names["subject_id"],
            "start": self.column_names["time"],
            "x": self.column_names["x"],
            "y": self.column_names["y"],
            "Avg_Pupil_Size_X": self.column_names["pupil_diameter"],
            "duration": self.column_names["duration"],
            "end": self.column_names["fixation_end"],
        }
        self.label = "Post_SumOfCorrect_NewSum"

    def prepare_files(self, file_references, metadata_references):
        labels = pd.Series()
        dataset = []
        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f)
        for file_reference in file_references:
            dataset, labels = self.prepare_file(
                file_reference, metadata_file, dataset, labels
            )
        return dataset, labels

    def prepare_file(self, file_reference, metadata_file, dataset, labels):
        subject_id = file_reference.reference.split("_")[-2]
        with file_reference.open("r") as f:
            csv = pd.read_csv(f)
        csv = csv.rename(columns=self.column_name_mapping)
        csv[self.column_names["subject_id"]] = int(subject_id)
        dataset.append(csv)
        labels.at[int(subject_id)] = (
            metadata_file[metadata_file["StudentID"] == int(subject_id)]
            .loc[:, self.label]
            .iat[0]
        )
        return dataset, labels

    def __str__(self):
        return super().__str__()
