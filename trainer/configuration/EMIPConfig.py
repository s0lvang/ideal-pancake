from trainer.configuration.DatasetConfig import DatasetConfig


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

    def __str__(self):
        return "EMIPConfig"
