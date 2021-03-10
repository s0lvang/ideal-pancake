from classifier.Experiment import Experiment
from classifier.datasets.Fractions import Fractions
from classifier.datasets.Jetris import Jetris
from classifier.datasets.EMIP import EMIP
from classifier.datasets.CSCW import CSCW
from classifier.utils import powerset, normalize_series
from comet_ml import Experiment as CometExperiment
from comet_ml import ExistingExperiment as CometExistingExperiment
import pandas as pd
from classifier import globals


class ExperimentManager:
    def __init__(self, dataset_names):
        self.datasets, self.labels = self.download_datasets(dataset_names)
        self.dataset_names = dataset_names

    def download_datasets(self, dataset_names):
        datasets = {}
        labelss = {}
        for dataset_name in dataset_names:
            dataset_class = get_dataset(dataset_name)
            dataset, labels = dataset_class.download_premade_features()
            datasets[dataset_name] = dataset
            labelss[dataset_name] = normalize_series(labels)
            print(labelss[dataset_name])
        return datasets, labelss

    def run_experiments(self):
        dataset_names = self.dataset_names[1:]
        dataset_combinations = powerset(dataset_names)
        for dataset_combination in dataset_combinations:
            self.run_experiment(dataset_combination)

        globals.comet_logger = CometExistingExperiment(
            api_key=globals.flags.comet_api_key,
            previous_experiment=globals.comet_logger.get_key(),
        )

    def run_experiment(self, dataset_combination):
        oos_dataset_name = self.dataset_names[0]
        comet_exp = CometExperiment(
            api_key=globals.flags.comet_api_key, project_name="ideal-pancake"
        )

        # Run experiment
        dataset, labels = self.merge_datasets(dataset_combination)
        experiment = Experiment(
            dataset=dataset,
            labels=labels,
            oos_dataset=self.datasets[oos_dataset_name],
            oos_labels=self.labels[oos_dataset_name],
        )
        metrics = experiment.run_experiment()

        # Logging
        comet_exp.set_name(globals.flags.experiment_name)
        comet_exp.log_metrics(metrics)
        comet_exp.log_other("in-study", dataset_combination)
        comet_exp.log_other("out-of-study", oos_dataset_name)

    def merge_datasets(self, dataset_combination):
        datasets = [self.datasets[dataset_name] for dataset_name in dataset_combination]
        labels = [self.labels[dataset_name] for dataset_name in dataset_combination]
        return pd.concat(datasets), pd.concat(labels)


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
