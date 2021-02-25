from feature_generation.datasets.CSCW import CSCW
from feature_generation.datasets.Fractions import Fractions
from feature_generation.datasets.Jetris import Jetris
from feature_generation.datasets.EMIP import EMIP
from feature_generation.datasets.MoocImages import MoocImages
from feature_generation.datasets.EMIPImages import EMIPImages


def init(dataset_name, experiment, _flags):
    global flags
    flags = _flags

    global dataset
    dataset = get_dataset(dataset_name=dataset_name)
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
    elif dataset_name == "fractions":
        return Fractions()
    elif dataset_name == "cscw":
        return CSCW()
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")
