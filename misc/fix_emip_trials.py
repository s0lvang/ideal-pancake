import os
import pandas as pd
from itertools import takewhile
import numpy as np
from pygazeanalyser import detectors


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headiterlist = [*headiter]
    length = len(headiterlist)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return headiterlist, length


def remove_nonimportant_messages(df):
    df = df[~df["L Raw X [px]"].astype("str").str.contains("UE-mouseclick")]
    return df


def mark_trials_and_calibration(df):
    messages = df[df["Type"] == "MSG"]
    df = df[df["Type"] == "SMP"]

    df["status"] = ""
    df["trial_number"] = 0
    trial_number = 0
    for i in range(len(messages.index) - 1):
        current_message = messages.iloc[i]
        next_message = messages.iloc[i + 1]
        status = ""
        if current_message["L Raw X [px]"] == "# Message: instruction_calibration.jpg":
            status = "CALIBRATION"
        elif (
            current_message["L Raw X [px]"]
            == "# Message: instruction_comprehension.jpg"
        ):
            status = "INSTRUCTING"
            trial_number += 1
        elif "mupliple" in current_message["L Raw X [px]"]:
            status = "TEST"
        else:
            status = "READING"
        current_message_time = current_message.loc["Time"]
        next_message_time = next_message.loc["Time"]
        df.loc[
            df["Time"].between(current_message_time, next_message_time),
            ["status", "trial_number"],
        ] = (status, trial_number)
    df.loc[
        df["Time"].between(next_message_time, df["Time"].iloc[-1]),
        ["status", "trial_number"],
    ] = (
        status,
        trial_number,
    )  # Get last value aswell
    return df


def rewrite_df(df):
    df = remove_nonimportant_messages(df)
    df = mark_trials_and_calibration(df)
    return df


def get_fixations(df):
    df = df[df["Type"] == "SMP"]
    fixations = detectors.fixation_detection(
        df["L POR X [px]"].to_numpy(),
        df["L POR Y [px]"].to_numpy(),
        df["Time"].to_numpy(),
        maxdist=25,
        mindur=500,
    )
    entries = [create_dataframe_entry(*fixation, df) for fixation in fixations[1]]
    return pd.DataFrame(entries)


def create_dataframe_entry(starttime, endtime, duration, endx, endy, df):
    df = df[df["Time"].between(starttime, endtime)]
    entry = {
        "fixationStart": starttime // 1000,
        "fixationEnd": endtime // 1000,
        "duration": duration // 1000,
        "x": df["L POR X [px]"].mean(),
        "y": df["L POR Y [px]"].mean(),
        "averagePupilSize": df["L Mapped Diameter [mm]"].mean(),
        "trial_number": df["trial_number"].mode()[0],
        "status": df["status"].mode()[0],
    }
    return entry


basepath = "../datasets/emip/data"
for file in os.listdir(basepath):
    with open(os.path.join(basepath, file)) as f:
        header, length = get_header(f)
        df = pd.read_csv(f, sep="\t", skiprows=length)
        rewritten_df = rewrite_df(df)
        fixations = get_fixations(rewritten_df)
        fixations.to_csv(
            f"../datasets/emip-fixations/data/{file}", sep="\t"
        )  # prepend the headers after write
    with open(f"../datasets/emip-fixations/data/{file}", "r+") as fr:
        content = fr.read()
        fr.seek(0, 0)
        fr.write("".join(header) + content)
