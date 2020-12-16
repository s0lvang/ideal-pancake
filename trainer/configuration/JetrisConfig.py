from trainer.configuration.DatasetConfig import DatasetConfig


class JetrisConfig(DatasetConfig):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "jetris"
        self.TSFRESH_FEATURES = {
            "length": None,
            "fft_aggregated": [
                {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
            ],
            "fft_coefficient": [{"coeff": k, "attr": "real"} for k in range(100)],
        }
        self.NUMERIC_FEATURES = [
            "Speed",
        ]
        self.CATEGORICAL_FEATURES = []
        self.FEATURE_COLUMNS = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES

    def __str__(self):
        return "JetrisConfig"