import numpy as np
import math

"""
The fixations contains information of the following saccade
"""


def get_saccade_length(data):
    coordinates = data.loc[:, ["x", "y"]].to_numpy()
    shifted_coordinates = np.roll(coordinates, -1, axis=0).tolist()
    shifted_coordinates[-1] = coordinates[-1]
    return [np.linalg.norm(a - b) for a, b in zip(coordinates, shifted_coordinates)]


def get_saccade_duration(data):
    endtimes = data.loc[:, "fixation_end"].to_numpy()
    starttimes = data.loc[:, "time"].to_numpy()
    shifted_starttimes = np.roll(starttimes, -1, axis=0).tolist()
    shifted_starttimes[-1] = starttimes[-1]
    saccades_durations = shifted_starttimes - endtimes
    saccades_durations[-1] = 0
    return saccades_durations


def get_angle_of_saccades(data):
    coordinates = data.loc[:, ["x", "y"]].to_numpy()
    shifted_coordinates = np.roll(coordinates, -1, axis=0).tolist()
    shifted_coordinates[-1] = coordinates[-1]  # dette funker ikke må settes
    return [
        angle_between_coordinates(a, b)
        for a, b in zip(coordinates, shifted_coordinates)
    ]


def angle_between_coordinates(a, b):
    radians = math.atan2(b[1] - a[1], b[0] - a[0])
    return abs(radians)
