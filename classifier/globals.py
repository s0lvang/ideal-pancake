from classifier.datasets.Jetris import Jetris
from classifier.datasets.EMIP import EMIP
from classifier.datasets.MoocImages import MoocImages
from classifier.datasets.EMIPImages import EMIPImages


def init(in_study, out_of_study, experiment, _flags):
    global flags
    flags = _flags

    global dataset
    dataset = get_dataset(dataset_name=in_study)
    global out_of_study_dataset
    if out_of_study:
        out_of_study_dataset = get_dataset(dataset_name=out_of_study)
    else:
        out_of_study_dataset = None
    global comet_logger
    comet_logger = experiment
    dataset_type = (
        str(dataset.__class__.__bases__[0]).split(".")[-1].strip(">'")
    )  # it gets the parentsclass name
    comet_logger.add_tag(dataset_type)


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
