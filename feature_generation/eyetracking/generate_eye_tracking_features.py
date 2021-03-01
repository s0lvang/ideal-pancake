from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import numpy as np


def generate_eye_tracking_features(data):
    return pd.concat([generate_features(subject) for subject in data])


def generate_features(subject):
    eye_tracking_features = pd.DataFrame()
    eye_tracking_features["information_processing_ratio"] = [
        get_information_processing_ratio(subject)
    ]
    return eye_tracking_features


def get_information_processing_ratio(subject):
    percentiles_saccade_length = np.percentile(subject.loc[:, "saccade_length"], 25)
    percentiles_fixation_duration = np.percentile(subject.loc[:, "duration"], 25)
    LIP = 0
    GIP = 0
    for saccade_length, fixation_duration in subject.loc[
        :, ["saccade_length", "duration"]
    ].to_numpy():
        fixation_is_short = fixation_duration < percentiles_fixation_duration
        saccade_is_short = saccade_length < percentiles_saccade_length
        if not fixation_is_short and saccade_is_short:
            LIP += 1
        elif fixation_is_short and not saccade_is_short:
            GIP += 1
        else:
            continue
    return GIP / LIP
