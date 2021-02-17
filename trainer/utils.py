import os
from comet_ml import ExistingExperiment, Experiment
import joblib
from tensorflow.io import gfile
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from trainer import globals
import argparse


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
    path = f"g://{output_path}"
    if not gfile.exists(path):
        gfile.makedirs(os.path.dirname(path))
    with gfile.GFile(path, "w") as wf:
        joblib.dump(object_to_dump, wf)


def download_object(path):
    bucket_path = f"g://{path}"
    obj = joblib.load(bucket_path)
    return obj


def log_hyperparameters_to_comet(clf):
    for i in range(len(clf.cv_results_["params"])):
        exp = Experiment(
            workspace="s0lvang",
            project_name="ideal-pancake",
            api_key=globals.flags.comet_api_key,
        )
        exp.add_tag("hp_tuning")
        exp.add_tags(globals.comet_logger.get_tags())
        for k, v in clf.cv_results_.items():
            if k == "params":
                exp.log_parameters(v[i])
            else:
                exp.log_metric(k, v[i])
        exp.end()

    old_experiment = ExistingExperiment(
        api_key=globals.flags.comet_api_key,
        previous_experiment=globals.comet_logger.get_key(),
    )
    globals.comet_logger = old_experiment


def log_dataframe_to_comet(df, name):
    globals.comet_logger.log_table(f"{name}.csv", tabular_data=df)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
