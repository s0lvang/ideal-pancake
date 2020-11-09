CSV_COLUMNS = None  # Schema of the data. Necessary for data stored in GCS

NUMERIC_FEATURES = [
    'R Mapped Diameter [mm]',
    'L Mapped Diameter [mm]',
    'Pupil Confidence',
]

CATEGORICAL_FEATURES = [
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL = 'expertise_programming'

METRIC_FILE_NAME = 'eval_metrics.joblib'
MODEL_FILE_NAME = 'model.joblib'

