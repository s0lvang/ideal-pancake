from trainer.configuration.JetrisConfig import JetrisConfig
from trainer.configuration.EMIPConfig import EMIPConfig
from trainer.configuration.MoocImagesConfig import MoocImagesConfig
from trainer.configuration.EMIPImagesConfig import EMIPImagesConfig


def init_emip_images():
    global config
    config = EMIPImagesConfig()


def init_mooc_images():
    global config
    config = MoocImagesConfig()


def init_emip():
    global config
    config = EMIPConfig()


def init_jetris():
    global config
    config = JetrisConfig()