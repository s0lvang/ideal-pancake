from trainer.configuration.DatasetConfig import DatasetConfig


class EMIPImagesConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.SUBJECT_ID_COLUMN = "id"
        self.LABEL = "expertise_programming"

    def __str__(self):
        return "EMIPImagesConfig"
