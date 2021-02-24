import os
import pandas as pd

column_name_mapping = {
    "Start_iViewX_micros": "start",
    "End_iViewX_micros": "end",
    "Location_X": "x",
    "Location_Y": "y",
    "Duration_micros": "duration",
}


def rewrite_df(df):
    df = df.rename(columns=column_name_mapping)
    df["duration"] = df["duration"] / 1000
    df["start"] = df["start"] / 1000
    df["end"] = df["end"] / 1000
    return df


basepath = "../datasets/fractions/data"
for file in os.listdir(basepath):
    with open(os.path.join(basepath, file)) as f:
        df = pd.read_csv(f, sep="\t")
        rewritten_df = rewrite_df(df)
        rewritten_df.to_csv(f"../datasets/fractions/data/{file}")
