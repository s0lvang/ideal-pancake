import os
from pathlib import Path
import arff
import pandas
from itertools import takewhile

directory = "datasets/emip_dataset/rawdata"
destination_directory = "datasets/emip_dataset/arff"

pathlist = Path(directory).rglob("*.tsv")


def get_comments(path):
    with open(path, "r") as fobj:
        headiter = takewhile(lambda s: s.startswith("##"), fobj)
        headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
        header = dict(filter(lambda x: len(x) == 2, headerList))
        split_on_tab = lambda x: x.split("\t")[1:]
        header = {k: split_on_tab(v) for k, v in header.items()}
    return header


def create_metadata(comments):
    metadata = f"""%@METADATA width_px {comments["Calibration Area"][0]} 
%@METADATA height_px {comments["Calibration Area"][1]} 
%@METADATA width_mm {comments["Stimulus Dimension [mm]"][0]}
%@METADATA height_mm {comments["Stimulus Dimension [mm]"][1]}
%@METADATA distance_mm {comments["Head Distance [mm]"][0]}
"""
    return metadata


def convert_tsv_to_arff(path):
    df = pandas.read_csv(path, comment="#", sep="\t")
    df["x"] = df[["R Raw X [px]", "L Raw X [px]"]].mean(axis=1)
    df["y"] = df[["R Raw Y [px]", "L Raw Y [px]"]].mean(axis=1)
    df["confidence"] = df[["R Validity", "L Validity"]].mean(axis=1)
    df = df.filter(["Time", "x", "y", "confidence"])
    destination = os.path.join(destination_directory, os.path.splitext(os.path.basename(path))[0] + ".arff")
    arff.dump(destination, df.values, relation="relation_name", names=df.columns)
    comments = get_comments(path)
    metadata = create_metadata(comments)
    with open(destination,"r") as f:
        data = f.read().replace("@attribute", "@ATTRIBUTE").replace("@relation","@RELATION").replace("@data","@DATA").replace("real", "NUMERIC")

    with open(destination,"w") as f: f.write(metadata+data)



for path in pathlist:
    path_in_str = str(path)
    convert_tsv_to_arff(path_in_str)
