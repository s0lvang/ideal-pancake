import os
from itertools import takewhile
import pandas as pd
import joblib
from tensorflow.io import gfile
from trainer import globals
from google.cloud import storage
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
import re
from trainer.datasets import datasets


def read_heatmaps():
    directory_name = "images"
    label_column = globals.dataset.config.LABEL
    metadata = pd.read_csv("emip/emip_metadata.csv/emip_metadata.csv")
    images = np.array([])
    labels = np.array([])
    subject_directories = os.listdir(directory_name)
    for subject_directory in subject_directories:
        subject_id = int(subject_directory)
        subject_directory = os.path.join(directory_name, subject_directory)
        print(subject_directory)
        frames_for_subjects = np.array(
            [
                cv2.resize(
                    cv2.imread(os.path.join(subject_directory, file)), (300, 170)
                )
                for file in sorted(os.listdir(subject_directory))
            ]
        )
        label = metadata.loc[metadata["id"] == int(subject_id), label_column]
        images = (
            np.concatenate((images, np.array([frames_for_subjects])))
            if images.size
            else np.array([frames_for_subjects])
        )
        labels = np.hstack((labels, label))
    return images, encode_labels(labels)


def encode_labels(labels):
    encoding = {"high": 3, "medium": 2, "low": 1, "none": 0}
    return np.array(list(map(lambda label: encoding[label.lower()], labels)))


def decode_labels(labels):
    encoding = ["none", "low", "medium", "high"]
    return list(map(lambda label: encoding[label], labels))


def preprocess_for_imagenet(dataset):
    print(dataset.shape)
    return np.array([preprocess_input(x) for x in dataset])


def upload_to_gcs(local_path, gcs_path):
    """Upload local file to Google Cloud Storage.

    Args:
      local_path: (string) Local file
      gcs_path: (string) Google Cloud Storage destination

    Returns:
      None
    """
    gfile.copy(local_path, gcs_path)


def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.

    Args:
      object_to_dump: Python object to be pickled
      output_path: (string) output path which can be Google Cloud Storage

    Returns:
      None
    """

    if not gfile.exists(output_path):
        gfile.makedirs(os.path.dirname(output_path))
    with gfile.open(output_path, "w") as wf:
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


def convert_labels_to_categorical(labels):
    average_score = sum(labels) / len(labels)
    return list(map(lambda score: "high" if (score > average_score) else "low", labels))
