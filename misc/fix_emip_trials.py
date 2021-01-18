import os
import pandas as pd
from itertools import takewhile
import numpy as np


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    length = len(list(headiter))
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header, length


def remove_nonimportant_messages(df):
    df = df[~df["L Raw X [px]"].astype("str").str.contains("UE-mouseclick")]
    return df


def mark_trials_and_calibration(df):
    messages = df[df["Type"] == "MSG"]
    # df = df[: (df["L Raw X [px]"] == messages.iloc[0, 4]).idxmax()]

    # Remove everything before the first message since it is uncalibrated noise

    df["status"] = ""
    df["trial_number"] = 1
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
        elif "mupliple" in current_message["L Raw X [px]"]:
            status = "TEST"
        else:
            status = "READING"
        print(status)
        current_message_time = current_message.loc["Time"]
        next_message_time = next_message.loc["Time"]
        df.loc[
            df["Time"].between(current_message_time, next_message_time),
            "status",
        ] = status
    df.loc[
        df["Time"].between(next_message_time, df["Time"].iloc[-1]), "status"
    ] = status  # Get last value aswell
    return df


def rewrite_df(df):
    df = remove_nonimportant_messages(df)
    df = mark_trials_and_calibration(df)
    return df


basepath = "../datasets/emip/data"
for file in os.listdir(basepath):
    with open(os.path.join(basepath, file)) as f:
        print(file)
        header, length = get_header(f)
        print(length)
        df = pd.read_csv(f, sep="\t", skiprows=length)
        rewritten_df = rewrite_df(df)
        rewritten_df.to_csv(
            f"../datasets/emip-rewritten/data/{file}"
        )  # remember to prepend the headers after write
