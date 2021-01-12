from trainer.configuration.JetrisConfig import JetrisConfig
from trainer.configuration.EMIPConfig import EMIPConfig
from trainer.configuration.MoocImagesConfig import MoocImagesConfig
from trainer.configuration.EMIPImagesConfig import EMIPImagesConfig
from trainer.Dataset import Dataset


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
    dataset = Dataset(in_study, get_config(in_study))
    global out_of_study_dataset
    if out_of_study:
        out_of_study_dataset = Dataset(out_of_study, get_config(out_of_study))
    else:
        out_of_study_dataset = None


def get_config(dataset_name):
    if dataset_name == "emip-images":
        return EMIPImagesConfig()
    elif dataset_name == "mooc-images":
        return MoocImagesConfig()
    elif dataset_name == "emip":
        return EMIPConfig()
    elif dataset_name == "jetris":
        return JetrisConfig()
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")