from trainer.datasets.Jetris import Jetris
from trainer.datasets.EMIP import EMIP
from trainer.datasets.MoocImages import MoocImages
from trainer.datasets.EMIPImages import EMIPImages


def init(in_study, out_of_study):
    global FORCE_GCS_DOWNLOAD
    FORCE_GCS_DOWNLOAD = False
    global FORCE_LOCAL_FILES
    FORCE_LOCAL_FILES = False
    global METRIC_FILE_NAME
    METRIC_FILE_NAME = "eval_metrics.joblib"
    global MODEL_FILE_NAME
    MODEL_FILE_NAME = "model.joblib"

    global dataset
    dataset = get_dataset(dataset_name=in_study)
    global out_of_study_dataset
    if out_of_study:
        out_of_study_dataset = get_dataset(dataset_name=out_of_study)
    else:
        out_of_study_dataset = None


def get_dataset(dataset_name):
    if dataset_name == "emip-images":
        return EMIPImages()
    elif dataset_name == "mooc-images":
        return MoocImages()
    elif dataset_name == "emip":
        return EMIP()
    elif dataset_name == "jetris":
        return Jetris()
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")
