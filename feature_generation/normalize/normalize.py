from feature_generation import globals
import pandas as pd
from feature_generation.eyetracking.saccades import (
    get_saccade_duration,
)
import numpy as np
from matplotlib import pyplot as plt


def normalize_data(data):
    return [normalize_columns(df) for df in data]


def normalize_columns(df):
    column_names = globals.dataset.column_names
    df[column_names["pupil_diameter"]] = min_max_normalize(
        df[column_names["pupil_diameter"]]
    )
    df = normalize_x_and_y(df)
    df = normalize_time(df)
    return df


def min_max_normalize(values):
    return (values - values.min()) / (values.max() - values.min())


def normalize_time(df):
    column_names = globals.dataset.column_names
    min_time = df[column_names["time"]].min()
    df[column_names["time"]] = df[column_names["time"]] - min_time
    df[column_names["fixation_end"]] = df[column_names["fixation_end"]] - min_time
    df = fix_outliers_in_time(df)
    return df


def normalize_x_and_y(df):
    df["x_normalized"] = min_max_normalize(df["x"]) * 1000
    df["y_normalized"] = min_max_normalize(df["y"]) * 1000
    return df


def fix_outliers_in_time(df):
    saccade_durations = pd.Series(get_saccade_duration(df))
    saccade_durations.index = df.index
    median_duration = saccade_durations.median()
    threshold = 1000
    bool_series = saccade_durations > threshold
    indices = df[bool_series].index
    for i in indices:
        diff = saccade_durations[i]
        df.loc[i + 1 :, "time"] -= diff - median_duration
        df.loc[i + 1 :, "fixation_end"] -= diff - median_duration
    return df
