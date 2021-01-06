from trainer.configuration.DatasetConfig import DatasetConfig
from trainer.datasets import heatmaps


class EMIPImagesConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.SUBJECT_ID_COLUMN = "id"
        self.LABEL = "age"
        self.DATASET_NAME = "emip-images"
        self.file_preparer = heatmaps.prepare_files

    def __str__(self):
        return super().__str__()