import json


class DatasetConfig:
    def __init__(self):
        self.FORCE_GCS_DOWNLOAD = False
        self.FORCE_LOCAL_FILES = False
        self.METRIC_FILE_NAME = "eval_metrics.joblib"
        self.MODEL_FILE_NAME = "model.joblib"

    LABEL = None

    # Time series
    NUMERIC_FEATURES = []
    CATEGORICAL_FEATURES = []
    FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    TSFRESH_FEATURES = {}

    experiment = None

    # timeseries_or_images = None
    # features = None
    # tsfresh_features = None
    # id_column_name = None

    # # time specific
    # time_column_name = None

    def __str__(self):
        variables = vars(self).copy()
        ts_fresh = ", ".join(variables.pop("TSFRESH_FEATURES", {}).keys())
        file_preparer = variables.pop("file_preparer")
        experimenter = variables.pop("experimenter")
        return f"{json.dumps(variables)} TS_FRESH_FEATURES: ({ts_fresh}) file_preparer: ({file_preparer.__name__}) experimenter: ({experimenter.__name__})"
