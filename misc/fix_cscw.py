import os
import pandas as pd

headers = [
    "Group",
    "Participant",
    "Fixation Start [ms]",
    "Fixation Duration [ms]",
    "Fixation End [ms]",
    "Position X",
    "Position Y",
    "Average Pupil Size [px] X",
    "Average Pupil Size [px] Y",
    "Dispersion X",
    "Dispersion Y",
    "Eye L/R",
    "Number",
    "AOI_Name",
    "AOI_geometry",
    "AOI_x1",
    "AOI_y1",
    "AOI_x2",
    "AOI_y2",
    "yolo",
    "yolo2",
    "yolo3",
    "yolo4",
    "yolo5",
]


basepath = "../datasets/cscw/data"
for file in os.listdir(basepath):
    if "video" in file.lower():
        with open(os.path.join(basepath, file)) as f:
            df = pd.read_csv(f)
            df = df.dropna(axis=1, how="all")
            df.columns = headers[0 : len(df.keys())]
            df.to_csv(f"../datasets/cscw/data/{file}")
