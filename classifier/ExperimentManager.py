from classifier.Experiment import Experiment
from classifier.datasets.Fractions import Fractions
from classifier.datasets.Jetris import Jetris
from classifier.datasets.EMIP import EMIP
from classifier.datasets.CSCW import CSCW
from classifier.Labels import Labels


class ExperimentManager:
    def __init__(self, dataset_names):
        self.datasets, self.labels = self.download_datasets(dataset_names)

    def download_datasets(self, dataset_names):
        datasets = {}
        labelss = {}
        for dataset_name in dataset_names:
            dataset_class = get_dataset(dataset_name)
            dataset, labels = dataset_class.download_premade_features()
            datasets[dataset_name] = dataset
            labelss[dataset_name] = Labels(labels, dataset_class.labels_are_categorical)
        return datasets, labelss

    def run_experiments(self):
        experiment = Experiment(
            list(self.datasets.values())[0],
            list(self.labels.values())[0],
            list(self.datasets.values())[1],
            list(self.labels.values())[1],
        )
        experiment.run_experiment()


def get_dataset(dataset_name):
    if dataset_name == "emip":
        return EMIP()
    elif dataset_name == "jetris":
        return Jetris()
    elif dataset_name == "fractions":
        return Fractions()
    elif dataset_name == "cscw":
        return CSCW()
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")