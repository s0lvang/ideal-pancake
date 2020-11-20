import pandas as pd
from google.cloud import storage
from itertools import takewhile
import os
import numpy as np
from misc.gazeheatplot import draw_heatmap
from misc.heatmap import Heatmapper
from PIL import Image


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header


def download_or_read_from_disk(blob):
    dataset_dir = os.path.join(blob.bucket.name, blob.name.split("/")[0])
    destination_file_name = os.path.join(dataset_dir, os.path.basename(blob.name))
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(destination_file_name):
        blob.download_to_filename(destination_file_name)
    return open(destination_file_name, "r")


def read_emip_from_gcs():
    bucket_name = "emip"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    dataset = []
    metadata_emip = None
    headers = []
    labels = pd.Series()
    blobs = list(bucket.list_blobs(delimiter="/"))
    files = filter(lambda file: "metadata" not in file.name, blobs)
    metadata_emip = next(filter(lambda blob: "metadata" in blob.name.lower(), blobs))
    with download_or_read_from_disk(metadata_emip) as f:
        metadata_emip = pd.read_csv(f)
    for blob in [*files]:
        with download_or_read_from_disk(blob) as f:
            header = get_header(f)
            csv = pd.read_csv(f, sep="\t", comment="#")
            dataset.append(csv)
            headers.append(header)
    return dataset, headers, metadata_emip


def mm_to_px(mm):
    dpi = 94
    return int(mm * dpi / 25.4)


def main():
    dataset, comments, metadata = read_emip_from_gcs()
    for data, comment in zip(dataset, comments):
        data["x"] = data[["R Raw X [px]", "L Raw X [px]"]].mean(axis=1)
        data["y"] = data[["R Raw Y [px]", "L Raw Y [px]"]].mean(axis=1)
        n = 1000
        data_chunks = [data[i : i + n] for i in range(0, data.shape[0], n)]
        directory = f"images/{comment['Subject'][0]}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for index, data_chunk in enumerate(data_chunks):
            output_name = f"images/{comment['Subject'][0]}/{index if index > 9 else '0'+str(index)}.png"
            background_image = "vehicle_java.jpg"
            data_chunk = data_chunk[data_chunk["y"].notna()]
            data_chunk = data_chunk[data_chunk["x"].notna()]
            gaze_data = [
                tuple(map(int, row)) for row in data_chunk[["x", "y"]].to_numpy()
            ]
            img = Image.new('RGB', (1000, 1200))
            heatmapper = Heatmapper()
            heatmap = heatmapper.heatmap_on_img(gaze_data, img)
            heatmap.save(output_name)


main()