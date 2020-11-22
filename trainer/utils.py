import os
from itertools import takewhile
import pandas as pd
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import metadata
from google.cloud import storage
import numpy as np
import cv2
from trainer.metadata import LABEL
from sklearn.preprocessing import OneHotEncoder


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header


def read_emip_from_gcs():
    bucket_name = "emip"
    directory_name = "test_folder/"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    dataset = pd.DataFrame()
    metadata_emip = None
    label_column = metadata.LABEL
    labels = pd.Series()
    blobs = list(bucket.list_blobs(delimiter="/"))
    files = filter(
        lambda file: file.name != directory_name and "metadata" not in file.name, blobs
    )
    metadata_emip = next(filter(lambda blob: "metadata" in blob.name.lower(), blobs))
    with download_or_read_from_disk(metadata_emip) as f:
        metadata_emip = pd.read_csv(f)
    for blob in [*files]:
        with download_or_read_from_disk(blob) as f:
            subject_id = get_header(f)["Subject"][0]
            csv = pd.read_csv(f, sep="\t", comment="#")
            csv["id"] = int(subject_id)
            dataset = dataset.append(csv, ignore_index=True)
            labels.at[int(subject_id)] = metadata_emip.loc[
                metadata_emip["id"] == int(subject_id) - 1, label_column
            ]
    return dataset, labels


def read_jetris_from_gcs():
    bucket_name = "jetris"
    directory_name = "game_raw/"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    dataset = pd.DataFrame()
    labels = []
    blobs = list(bucket.list_blobs(delimiter="/", prefix=directory_name))
    files = filter(lambda file: file.name != directory_name, blobs)
    for blob in files:
        with download_or_read_from_disk(blob) as f:
            csv = pd.read_csv(f, comment="#")
            csv = csv[
                csv["Pupil.initial"] != "saccade"
            ]  # this drops all lines that are saccades, we should do something smarter here.
            game_id = csv["gameID"][0]
            new_row = {k: csv[k].fillna(method="ffill") for k in csv.keys()}
            dataset = dataset.append(new_row, ignore_index=True)
            labels.append(csv["Score.1"].iloc[-1])
    average_score = sum(labels) / len(labels)
    print(average_score)
    categorical_labels = list(
        map(lambda score: "high" if (score > average_score) else "low", labels)
    )
    return dataset, np.array(categorical_labels)


def download_or_read_from_disk(blob):
    dataset_dir = os.path.join(blob.bucket.name, blob.name.split("/")[0])
    destination_file_name = os.path.join(dataset_dir, os.path.basename(blob.name))
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(destination_file_name):
        blob.download_to_filename(destination_file_name)
    return open(destination_file_name, "r")


def read_heatmaps():
    directory_name = "images"
    label_column = LABEL
    metadata = pd.read_csv("emip/emip_metadata.csv/emip_metadata.csv")
    images = np.array([])
    labels = np.array([])
    subject_directories = os.listdir(directory_name)
    for subject_directory in subject_directories[0:20]:
        subject_id = int(subject_directory)
        print(subject_id)
        subject_directory = os.path.join(directory_name, subject_directory)
        print(subject_directory)
        print(os.listdir(subject_directory))
        frames_for_subjects = np.array(
            [
                cv2.imread(os.path.join(subject_directory, file))
                for file in os.listdir(subject_directory)
            ]
        )
        print(frames_for_subjects.shape)
        label = metadata.loc[metadata["id"] == int(subject_id), label_column]
        images = (
            np.concatenate((images, np.array([frames_for_subjects])))
            if images.size
            else np.array([frames_for_subjects])
        )
        print(images.shape)
        labels = np.hstack((labels, label))
    return images, one_hot_encode_labels(labels)


def one_hot_encode_labels(labels):
    encoding = {"high": 3, "medium": 2, "low": 1, "none": 0}
    encoded_labels = list(map(lambda label: encoding[label.lower()], labels))
    print(encoded_labels)
    return np.eye(len(encoding.keys()))[encoded_labels]


def upload_to_gcs(local_path, gcs_path):
    """Upload local file to Google Cloud Storage.

    Args:
      local_path: (string) Local file
      gcs_path: (string) Google Cloud Storage destination

    Returns:
      None
    """
    gfile.Copy(local_path, gcs_path)


def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.

    Args:
      object_to_dump: Python object to be pickled
      output_path: (string) output path which can be Google Cloud Storage

    Returns:
      None
    """

    if not gfile.Exists(output_path):
        gfile.MakeDirs(os.path.dirname(output_path))
    with gfile.Open(output_path, "w") as wf:
        joblib.dump(object_to_dump, wf)


def boolean_mask(columns, target_columns):
    """Create a boolean mask indicating location of target_columns in columns.

    Args:
      columns: (List[string]), list of all columns considered.
      target_columns: (List[string]), columns whose position
        should be masked as 1.

    Returns:
      List[bool]
    """
    target_set = set(target_columns)
    return [x in target_set for x in columns]
