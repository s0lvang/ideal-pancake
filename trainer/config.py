CSV_COLUMNS = None  # Schema of the data. Necessary for data stored in GCS

NUMERIC_FEATURES = [
    "Pupil Confidence",
]

CATEGORICAL_FEATURES = []

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL = "expertise_programming"

METRIC_FILE_NAME = "eval_metrics.joblib"
MODEL_FILE_NAME = "model.joblib"

TSFRESH_FEATURES = {
    "length": None,
    "fft_aggregated": [
        {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
    ],
    "fft_coefficient": [{"coeff": k, "attr": "real"} for k in range(100)],
}
AVAILABLE_HEATMAP_DATASETS = ["emip-images", "mooc-images"]
AVAILABLE_TS_DATASETS = ["jetris", "emip"]

DATASET_NAME = "jetris"

FORCE_LOCAL_FILES = False

FORCE_GCS_DOWNLOAD = True
