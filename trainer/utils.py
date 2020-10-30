# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hold utility functions."""

import os
from itertools import takewhile
import pandas as pd
from sklearn import model_selection
import joblib
import tensorflow.compat.v1.gfile as gfile
from trainer import metadata
from google.cloud import storage
import numpy as np


def data_train_test_split(dataset, labels):
    """Split the DataFrame two subsets for training and testing.

    Args:
      dataset: [(df, label)]

    Returns:
     x_train list of dataframes of eyetracking data with columns selected
     y_train list of labels
     x_test dataframe of eyetracking data with columns selected
     y_test list of labels
    """

    columns_to_use = metadata.FEATURE_COLUMNS
    dataset_with_columns_to_use = dataset[columns_to_use]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        dataset_with_columns_to_use, labels
    )
    return x_train, y_train, x_test, y_test


def get_header(file):
    headiter = takewhile(lambda s: s.startswith("##"), file)
    headerList = list(map(lambda x: x.strip("##").strip().split(":"), headiter))
    header = dict(filter(lambda x: len(x) == 2, headerList))
    split_on_tab = lambda x: x.split("\t")[1:]
    header = {k: split_on_tab(v) for k, v in header.items()}
    file.seek(0, 0)
    return header


def read_emip_from_gcs():
    file_pattern = "g://emip/test_folder"
    # Download the files to local /tmp/ folder
    bucket_name = "emip"
    prefix = "test_folder/"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    dataset = pd.DataFrame()
    metadata_emip = None
    label_column = metadata.LABEL
    labels = [] 
    blobs = list(bucket.list_blobs(delimiter="/"))
    files = filter(
        lambda file: file.name != prefix and "metadata" not in file.name, blobs
    )
    metadata_emip = next(filter(lambda blob: "metadata" in blob.name.lower(), blobs))
    with download_or_read_from_disk(metadata_emip) as f:
        metadata_emip = pd.read_csv(f)
    for blob in [*files]:
        with download_or_read_from_disk(blob) as f:
            subject_id = get_header(f)["Subject"][0]
            csv = pd.read_csv(f, sep="\t", comment="#")
            new_row = {k: csv[k].fillna(method="ffill") for k in csv.keys()}
            dataset = dataset.append(new_row, ignore_index=True)
            labels.append(metadata_emip.loc[int(subject_id)-1, label_column])
            print(labels)
    return dataset, np.array(labels)


def download_or_read_from_disk(blob):
    dataset_dir = os.path.join(blob.bucket.name, blob.name.split("/")[0])
    destination_file_name = os.path.join(dataset_dir, os.path.basename(blob.name))
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isfile(destination_file_name):
        blob.download_to_filename(destination_file_name)
    return open(destination_file_name, "r")


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
