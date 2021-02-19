from classifier.datasets.Heatmap import Heatmap


class MoocImages(Heatmap):
    def __init__(self):
        super().__init__("mooc-images")
        self.subject_id_column = "subject"
        self.label = "posttest"
        self.labels_are_categorical = False

    def heatmap_label(self, metadata_file, id):
        label = metadata_file[metadata_file[self.subject_id_column] == id][self.label]
        return int(label)

    def subject_id(self, file_reference):
        return int(file_reference.reference.split("/")[-2])

    def __str__(self):
        return super().__str__()
