from trainer.configuration.DatasetConfig import DatasetConfig


class MoocImagesConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.SUBJECT_ID_COLUMN = "id"
        self.LABEL = "posttest"
        self.DATASET_NAME = "mooc-images"

    def __str__(self):
        return "MoocImagesConfig"
