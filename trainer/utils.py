import os
import joblib
from tensorflow.io import gfile
from keras.applications.imagenet_utils import preprocess_input
import argparse
import numpy as np


def preprocess_for_imagenet(dataset):
    return np.array([preprocess_input(x) for x in dataset])


def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.

    Args:
      object_to_dump: Python object to be pickled
      output_path: (string) output path which can be Google Cloud Storage

    Returns:
      None
    """
    path = f"gs://{output_path}"
    if not gfile.exists(path):
        gfile.makedirs(os.path.dirname(path))
    with gfile.GFile(path, "w") as wf:
        joblib.dump(object_to_dump, wf)


def download_object(path):
    bucket_path = f"gs://{path}"
    with gfile.GFile(bucket_path, "rb") as wf:
        obj = joblib.load(wf)
    return obj


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", ""):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
