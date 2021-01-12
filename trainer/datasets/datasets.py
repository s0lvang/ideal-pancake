import os
from itertools import takewhile
import pandas as pd
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import globals
from trainer.FileRefence import FileReference
from google.cloud import storage
import numpy as np


def datasets_and_labels(dataset_config):
    valid_config()
    file_references = get_file_references("data/")
    metadata_references = get_file_references("metadata/")
    datasets, labels = dataset_config.file_preparer(
        file_references, metadata_references
    )
    return datasets, labels


def valid_config():
    valid_download_settings()


def valid_download_settings():
    if globals.config.FORCE_LOCAL_FILES and globals.config.FORCE_GCS_DOWNLOAD:
        raise ValueError(
            "Both force_local_files and force_gcs_download cannot be true at the same time."
        )


def get_file_references(directory_name):
    if globals.config.FORCE_LOCAL_FILES:
        file_references = get_file_names_from_directory(
            f"datasets/{globals.config.DATASET_NAME}/{directory_name}"
        )
    else:
        file_references = get_blobs_from_gcs(
            bucket_name=globals.config.DATASET_NAME, prefix=directory_name
        )
    return file_references


def get_file_names_from_directory(directory_name):
    return recursive_file_names_from_dir(directory_name, [])


def recursive_file_names_from_dir(path, paths):
    if os.path.isdir(path):
        for sub_path in os.listdir(path):
            recursive_file_names_from_dir(os.path.join(path, sub_path), paths)
        return paths
    elif os.path.isfile(path):
        paths.append(FileReference(path))
        return paths
    else:
        raise ValueError(f"Got a path that isn't dir or file: {path}")


def get_blobs_from_gcs(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_references = list(
        map(FileReference, filter(lambda file: file.name != prefix, blobs))
    )
    return file_references
