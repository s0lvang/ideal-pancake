class DatasetConfig:
    def __init__(self):
        self.FORCE_GCS_DOWNLOAD = False
        self.FORCE_LOCAL_FILES = True
        self.METRIC_FILE_NAME = "eval_metrics.joblib"
        self.MODEL_FILE_NAME = "model.joblib"

    LABEL = None

    # Time series
    NUMERIC_FEATURES = []
    CATEGORICAL_FEATURES = []
    FEATURE_COLUMNS = []

    TSFRESH_FEATURES = {}

    experiment = None

    # timeseries_or_images = None
    # features = None
    # tsfresh_features = None
    # id_column_name = None

    # # time specific
    # time_column_name = None
