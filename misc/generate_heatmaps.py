import pandas as pd
from google.cloud import storage
from itertools import takewhile
import os
from heatmappy.heatmap import Heatmapper
from PIL import Image


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header


def read_emip_from_gcs():
    directory_name = "../datasets/emip-enhanced/data"
    dataset = []
    headers = []

    for file in os.listdir(directory_name):
        with open(os.path.join(directory_name, file)) as f:
            header = get_header(f)
            csv = pd.read_csv(f, sep="\t", comment="#")
            dataset.append(csv)
            headers.append(header)
    return dataset, headers


def average_if_not_zero(left, right):
    if right > 0 and left > 0:
        return (left + right) / 2
    elif right > 0:
        return right
    return left


def preprocess_data(data, trial):
    data = data[data["status"] == "READING"]
    data = data[data["trial_number"] == trial]
    data = data[["L POR X [px]", "R POR X [px]", "R POR Y [px]", "L POR Y [px]"]]

    # Take a conditional average of left eye and right eye
    data["x"] = data.apply(
        lambda x: average_if_not_zero(x["R POR X [px]"], x["L POR X [px]"]), axis=1
    )
    data["y"] = data.apply(
        lambda x: average_if_not_zero(x["R POR Y [px]"], x["L POR Y [px]"]), axis=1
    )

    # Remove nan_values
    data = data[data["x"].notna()]
    data = data[data["y"].notna()]

    # Normalize columns in range 1000
    """ data["x"] = (
        (data["x"] - data["x"].min()) / (data["x"].max() - data["x"].min())
    ) * 1000
    data["y"] = (
        (data["y"] - data["y"].min()) / (data["y"].max() - data["y"].min())
    ) * 1000
    """
    return data[["x", "y"]]


def main():
    dataset, comments = read_emip_from_gcs()
    number_of_frames = 54
    for data, comment in zip(dataset, comments):
        for i in range(1, 3):
            processed_data = preprocess_data(data, i)
            n = len(processed_data) // number_of_frames
            frames = [
                processed_data[i : i + n] for i in range(0, processed_data.shape[0], n)
            ]
            if len(frames) == 31:
                frames = frames[:-1]
            directory = f"images/{comment['Subject'][0]}_{i}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            for index, frame in enumerate(frames):
                output_name = (
                    f"{directory}/{index if index > 9 else '0'+str(index)}.png"
                )
                print(output_name)
                gaze_data = [tuple(map(int, row)) for row in frame.to_numpy()]
                img = Image.new("RGB", (1920, 1080))
                # img = Image.open("vehicle_java2.jpg")

                heatmapper = Heatmapper()
                heatmap = heatmapper.heatmap_on_img(gaze_data, img)
                heatmap = heatmap.resize((640, 360))
                heatmap.save(output_name)


main()
