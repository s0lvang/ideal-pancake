from trainer.configuration.DatasetConfig import DatasetConfig
from trainer.datasets import emip
from trainer import experiment
from trainer.Dataset import Dataset


class EMIPConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "emip"
        self.LABEL = "expertise_programming"
        self.TSFRESH_FEATURES = {
            "length": None,
            "fft_aggregated": [
                {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
            ],
            "fft_coefficient": [{"coeff": k, "attr": "real"} for k in range(100)],
        }
        self.NUMERIC_FEATURES = [
            "Pupil Confidence",
        ]
        self.CATEGORICAL_FEATURES = []
        self.FEATURE_COLUMNS = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        self.file_preparer = emip.prepare_emip_files
        self.experimenter = experiment.run_ts_experiment

    def __str__(self):
        return super().__str__()
