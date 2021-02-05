import os
import joblib
from tensorflow.io import gfile
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from collections import Counter
from random import uniform


def preprocess_for_imagenet(dataset):
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
