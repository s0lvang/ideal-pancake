from trainer.configuration.JetrisConfig import JetrisConfig
from trainer.configuration.EMIPConfig import EMIPConfig
from trainer.configuration.MoocImagesConfig import MoocImagesConfig
from trainer.configuration.EMIPImagesConfig import EMIPImagesConfig


def init(in_study, out_of_study):
    global config
    config = get_config(in_study)
    global out_of_study_config
    if out_of_study:
        out_of_study_config = get_config(out_of_study)
    else:
        out_of_study_config = None


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