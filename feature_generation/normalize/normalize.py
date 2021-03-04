from feature_generation import globals
from matplotlib import pyplot as plt


def normalize_data(data):
    return [normalize_columns(df) for df in data]


def normalize_columns(df):
    column_names = globals.dataset.column_names
    df[column_names["pupil_diameter"]] = normalize_pupil_diameter(
        df[column_names["pupil_diameter"]]
    )
    df = normalize_time(df)
    return df


def normalize_pupil_diameter(pupil_diameter):
    return (pupil_diameter - pupil_diameter.min()) / (
        pupil_diameter.max() - pupil_diameter.min()
    )


def normalize_time(df):
    column_names = globals.dataset.column_names
    min_time = df[column_names["time"]].min()
    df[column_names["time"]] = df[column_names["time"]] - min_time
    df[column_names["fixation_end"]] = df[column_names["fixation_end"]] - min_time
    return df