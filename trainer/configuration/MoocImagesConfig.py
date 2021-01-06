from trainer.configuration.DatasetConfig import DatasetConfig
from trainer.datasets import heatmaps


class MoocImagesConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.SUBJECT_ID_COLUMN = "subject"
        self.LABEL = "posttest"
        self.DATASET_NAME = "mooc-images"
        self.file_preparer = heatmaps.prepare_files

    def __str__(self):
        return super().__str__()