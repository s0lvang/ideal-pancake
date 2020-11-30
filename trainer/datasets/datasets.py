import os
from itertools import takewhile
import pandas as pd
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import config
from trainer.datasets import jetris, emip
from google.cloud import storage
import numpy as np


def datasets_and_labels():
    valid_config()
    file_references = get_file_references("data/")
    metadata_reference = get_file_references("metadata/")
    datasets, labels = prepare_files(file_references, metadata_reference)
    return datasets, labels


def prepare_files(file_references, metadata_reference):
    if config.DATASET_NAME is "jetris":
        return jetris.prepare_jetris_files(file_references)
    elif config.DATASET_NAME is "emip":
        return emip.prepare_emip_files(file_references, metadata_reference)


def valid_config():
    valid_dataset()
    valid_download_settings()


def valid_download_settings():
    if config.FORCE_LOCAL_FILES and config.FORCE_GCS_DOWNLOAD:
        raise ValueError(
            "Both force_local_files and force_gcs_download cannot be true at the same time."
        )


def valid_dataset():
    if config.DATASET_NAME not in config.AVAILABLE_DATASETS:
        raise ValueError(
            f"{config.DATASET_NAME} does not exist in {config.AVAILABLE_DATASETS}"
        )
    else:
        return True


def get_file_references(data_context, directory_name):
    if config.FORCE_LOCAL_FILES:
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


def get_files(file_reference):
    if config.FORCE_LOCAL_FILES:
        return open(file_reference, "r")
    else:
        return cached_download_data(file_reference)


def cached_download_data(blob):
    dataset_dir = os.path.join(blob.bucket.name, blob.name.split("/")[0])
    destination_file_name = os.path.join(dataset_dir, os.path.basename(blob.name))
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(destination_file_name) or config.FORCE_GCS_DOWNLOAD:
        blob.download_to_filename(destination_file_name)
    return open(destination_file_name, "r")
