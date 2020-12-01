import os
from itertools import takewhile
import pandas as pd
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import config
from trainer.datasets import jetris, emip, heatmaps
from trainer.FileRefence import FileReference
from google.cloud import storage
import numpy as np


def datasets_and_labels():
    valid_config()
    file_references = get_file_references("data/")
    metadata_references = get_file_references("metadata/")
    datasets, labels = prepare_files(file_references, metadata_references)
    return datasets, labels


def prepare_files(file_references, metadata_references):
    if config.DATASET_NAME == "jetris":
        return jetris.prepare_jetris_files(file_references)
    elif config.DATASET_NAME == "emip":
        return emip.prepare_emip_files(file_references, metadata_references)
    elif config.DATASET_NAME == "mooc-images":
        return heatmaps.prepare_files(
            file_references,
            metadata_references,
            config.MOOC_IMAGES_LABEL,
            config.MOOC_SUBJECT_ID_COLUMN,
        )
    elif config.DATASET_NAME == "emip-images":
        return heatmaps.prepare_files(
            file_references,
            metadata_references,
            config.MOOC_IMAGES_LABEL,
            config.MOOC_SUBJECT_ID_COLUMN,
        )


def valid_config():
    valid_download_settings()


def valid_download_settings():
    if config.FORCE_LOCAL_FILES and config.FORCE_GCS_DOWNLOAD:
        raise ValueError(
            "Both force_local_files and force_gcs_download cannot be true at the same time."
        )


def get_file_references(directory_name):
    if config.FORCE_LOCAL_FILES:
        file_references = get_file_names_from_directory(
            f"{config.DATASET_NAME}/{directory_name}"
        )
    else:
        file_references = get_blobs_from_gcs(
            bucket_name=config.DATASET_NAME, prefix=directory_name
        )
    return file_references


def get_file_names_from_directory(directory_name):
    file_names = [
        FileReference(f"{directory_name}{file_name}")
        for file_name in os.listdir(directory_name)
        if os.path.isfile(os.path.join(directory_name, file_name))
    ]
    return file_names


def get_blobs_from_gcs(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_references = list(
        map(FileReference, filter(lambda file: file.name != prefix, blobs))
    )
    return file_references
