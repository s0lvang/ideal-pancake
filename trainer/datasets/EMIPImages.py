from trainer.datasets.Heatmap import Heatmap


class EMIPImages(Heatmap):
    def __init__(self):
        super().__init__("emip-images-54-frames")
        self.subject_id_column = "id"
        self.label = "expertise_programming"
        self.labels_are_categorical = True

    def heatmap_label(self, metadata_file, id):
        label = metadata_file[
            metadata_file[self.subject_id_column] == int(id.split("_")[0])
        ][self.label]
        encoding = {"high": 3, "medium": 2, "low": 1, "none": 0}
        encoded_label = encoding[label.iloc[0]]
        return encoded_label

    def subject_id(self, file_reference):
        return file_reference.reference.split("/")[-2]

    def __str__(self):
        return super().__str__()
