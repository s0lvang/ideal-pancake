from numpy.lib.function_base import percentile
import math
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax


def generate_eye_tracking_features(data):
    return pd.concat([generate_features(subject) for subject in data])


def generate_features(subject):
    eye_tracking_features = pd.DataFrame()
    eye_tracking_features["information_processing_ratio"] = [
        get_information_processing_ratio(subject)
    ]
    eye_tracking_features["saccade_speed_skewness"] = get_saccade_speed_skewness(
        subject
    )
    eye_tracking_features["entropy_xy"] = get_entropy(subject)
    eye_tracking_features["saccade_verticality"] = get_horizontallity_of_saccades(
        subject
    )
    return eye_tracking_features


def get_saccade_speed_skewness(subject):
    saccade_speed = (
        subject.loc[:, "saccade_length"] / subject.loc[:, "saccade_duration"]
    )
    return saccade_speed.skew()


def get_horizontallity_of_saccades(subject):
    angles = subject.loc[:, "angle_of_saccades"]
    sinus_values = angles.apply(math.sin)
    return sinus_values.mean()


def get_entropy(subject):
    x = subject.loc[:, "x_normalized"]
    y = subject.loc[:, "y_normalized"]
    xedges = [i for i in range(0, int(x.max()), 50)]
    yedges = [i for i in range(0, int(y.max()), 50)]
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    return entropy(H.flatten())


def get_information_processing_ratio(subject):
    upper_threshold_saccade_length = np.percentile(subject.loc[:, "saccade_length"], 75)
    lower_threshold_saccade_length = np.percentile(subject.loc[:, "saccade_length"], 25)
    upper_threshold_fixation_duration = np.percentile(subject.loc[:, "duration"], 75)
    lower_threshold_fixation_duration = np.percentile(subject.loc[:, "duration"], 25)
    LIP = 0
    GIP = 0
    for saccade_length, fixation_duration in subject.loc[
        :, ["saccade_length", "duration"]
    ].to_numpy():
        fixation_is_short = fixation_duration <= lower_threshold_fixation_duration
        fixation_is_long = upper_threshold_fixation_duration <= fixation_duration
        saccade_is_short = saccade_length <= lower_threshold_saccade_length
        saccade_is_long = upper_threshold_saccade_length <= saccade_length
        if fixation_is_long and saccade_is_short:
            LIP += 1
        elif fixation_is_short and saccade_is_long:
            GIP += 1
        else:
            continue
    return GIP / (LIP + 1)
