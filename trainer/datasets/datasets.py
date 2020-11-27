import os
from itertools import takewhile
import pandas as pd
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import metadata
from google.cloud import storage
import numpy as np


def datasets_and_labels(dataset, force_local_files, force_gcs_download):
    if force_local_files and force_gcs_download:
        raise ValueError(
            "Both force_local_files and force_gcs_download cannot be true at the same time."
        )
    dataset_is_valid(dataset)
    file_references = get_file_references(force_local_files, dataset, "data/")
    metadata_reference = get_file_references(force_local_files, dataset, "metadata/")
    datasets, labels = prepare_files(
        dataset,
        file_references,
        metadata_reference,
        force_local_files,
        force_gcs_download,
    )
    return datasets, labels


def prepare_files(
    dataset, file_references, metadata_reference, force_local_files, force_gcs_download
):
    if dataset is "jetris":
        return prepare_jetris_files(
            file_references, force_local_files, force_gcs_download
        )
    elif dataset is "emip":
        return prepare_emip_files(
            file_references, metadata_reference, force_local_files, force_gcs_download
        )


def dataset_is_valid(dataset):
    if dataset not in metadata.AVAILABLE_DATASETS:
        raise ValueError(f"{dataset} does not exist in {metadata.available_datasets}")
    else:
        return True


def get_file_references(force_local_files, data_context, directory_name):
    if force_local_files:
        file_references = get_file_names_from_directory(
            f"{data_context}/{directory_name}"
        )
    else:
        file_references = get_blobs_from_gcs(
            bucket_name=data_context, directory_name=directory_name
        )
    return file_references


def get_file_names_from_directory(directory_name):
    file_names = [
        f"{directory_name}{file_name}"
        for file_name in os.listdir(directory_name)
        if os.path.isfile(os.path.join(directory_name, file_name))
    ]
    return file_names


def get_blobs_from_gcs(bucket_name, directory_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(delimiter="/", prefix=directory_name))
    file_references = list(filter(lambda file: file.name != directory_name, blobs))
    return file_references


def get_metadata_blob_from_gcs(bucket_name, directory_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(delimiter="/", prefix="metadata"))
    return blobs


def prepare_jetris_files(file_references, force_local_files, force_gcs_download):
    labels = pd.Series()
    dataset = pd.DataFrame()
    for file_reference in file_references:
        with get_files(file_reference, force_local_files, force_gcs_download) as f:
            dataset, labels = prepare_jetris_file(f, dataset, labels)
    # labels = convert_labels_to_categorical()
    dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
    return dataset, labels


def prepare_emip_files(
    file_references, metadata_references, force_local_files, force_gcs_download
):
    labels = pd.Series()
    dataset = pd.DataFrame()
    with get_files(metadata_references[0], force_local_files, force_gcs_download) as f:
        metadata_file = pd.read_csv(f)
    for file_reference in file_references:
        with get_files(file_reference, force_local_files, force_gcs_download) as f:
            dataset, labels = prepare_emip_file(f, metadata_file, dataset, labels)
    # dataset = dataset.rename(columns={"gameID": "id", "time[milliseconds]": "Time"})
    return dataset, labels


def prepare_emip_file(f, metadata_file, dataset, labels):
    subject_id = get_header(f)["Subject"][0]
    csv = pd.read_csv(f, sep="\t", comment="#")
    csv["id"] = int(subject_id)
    dataset = dataset.append(csv, ignore_index=True)
    labels.at[int(subject_id)] = metadata_file.loc[int(subject_id) - 1, metadata.LABEL]
    return dataset, labels


def get_files(file_reference, force_local_files, force_gcs_download):
    if force_local_files:
        return open(file_reference, "r")
    else:
        return cached_download_data(file_reference, force_gcs_download)


def prepare_jetris_file(f, dataset, labels):
    csv = pd.read_csv(f, comment="#")
    csv = csv[
        csv["Pupil.initial"] != "saccade"
    ]  # this drops all lines that are saccades, we should do something smarter here.
    game_id = csv["gameID"][0]
    dataset = dataset.append(csv, ignore_index=True)
    labels.at[int(game_id)] = csv["Score.1"].iloc[-1]
    return dataset, labels


def convert_labels_to_categorical(labels):
    average_score = sum(labels) / len(labels)
    categorical_labels = list(
        map(lambda score: "high" if (score > average_score) else "low", labels)
    )
    return categorical_labels


def cached_download_data(blob, force_gcs_download):
    dataset_dir = os.path.join(blob.bucket.name, blob.name.split("/")[0])
    destination_file_name = os.path.join(dataset_dir, os.path.basename(blob.name))
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(destination_file_name) or force_gcs_download:
        blob.download_to_filename(destination_file_name)
    return open(destination_file_name, "r")