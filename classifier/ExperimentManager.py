import itertools
from classifier.Experiment import Experiment
from classifier.datasets.Fractions import Fractions
from classifier.datasets.Jetris import Jetris
from classifier.datasets.EMIP import EMIP
from classifier.datasets.CSCW import CSCW
from classifier.utils import combine_regexes_and_filter_df, powerset, normalize_series
from comet_ml import Experiment as CometExperiment
from comet_ml import ExistingExperiment as CometExistingExperiment
import pandas as pd
from classifier import globals
import numpy as np
import random
import time

feature_group_regexes = [
    [
        "information_processing_ratio",
        "saccade_speed_skewness",
        "entropy_xy",
        "saccade_verticality",
    ],
    [
        "heatmaps_*",
    ],
    [
        "pupil_diameter_rolling__fft_aggregated_*",
        "^duration_rolling__fft_aggregated__*",
        "saccade_length_rolling__fft_aggregated__*",
        "saccade_duration_rolling__fft_aggregated__*",
    ],
    [
        "pupil_diameter_rolling__lhipa",
        "saccade_length_rolling__lhipa",
        "saccade_duration_rolling__lhipa",
        "^duration_rolling__lhipa",
    ],
    [
        "^duration_rolling__markov",
        "pupil_diameter_rolling__markov",
        "saccade_duration_rolling__markov",
        "saccade_length_rolling__markov",
    ],
    [
        "pupil_diameter_rolling__arma__*",
        "^duration_rolling__arma__*",
        "saccade_length_rolling__arma__*",
        "saccade_duration_rolling__arma__*",
    ],
    [
        "saccade_duration_rolling__garch",
        "^duration_rolling__garch",
        "pupil_diameter_rolling__garch",
        "saccade_length_rolling__garch",
    ],
    [
        "pupil_diameter_rolling__fft_aggregated_*",
        "pupil_diameter_rolling__lhipa",
        "pupil_diameter_rolling__markov",
        "pupil_diameter_rolling__arma__*",
        "pupil_diameter_rolling__garch",
    ],
    [
        "^duration_rolling__fft_aggregated__*",
        "^duration_rolling__lhipa",
        "^duration_rolling__markov",
        "^duration_rolling__arma__*",
        "^duration_rolling__garch",
    ],
    [
        "saccade_length_rolling__fft_aggregated__*",
        "saccade_length_rolling__lhipa",
        "saccade_length_rolling__markov",
        "saccade_length_rolling__arma__*",
        "saccade_length_rolling__garch",
    ],
    [
        "saccade_duration_rolling__fft_aggregated__*",
        "saccade_duration_rolling__lhipa",
        "saccade_duration_rolling__markov",
        "saccade_duration_rolling__arma__*",
        "saccade_duration_rolling__garch",
    ],
    [
        "information_processing_ratio",
        "saccade_speed_skewness",
        "entropy_xy",
        "saccade_verticality",
        "heatmaps_*",
        "pupil_diameter_rolling__fft_aggregated_*",
        "pupil_diameter_rolling__lhipa",
        "pupil_diameter_rolling__markov",
        "pupil_diameter_rolling__arma__*",
        "pupil_diameter_rolling__garch",
        "^duration_rolling__fft_aggregated__*",
        "^duration_rolling__lhipa",
        "^duration_rolling__markov",
        "^duration_rolling__arma__*",
        "^duration_rolling__garch",
        "saccade_length_rolling__fft_aggregated__*",
        "saccade_length_rolling__lhipa",
        "saccade_length_rolling__markov",
        "saccade_length_rolling__arma__*",
        "saccade_length_rolling__garch",
        "saccade_duration_rolling__fft_aggregated__*",
        "saccade_duration_rolling__lhipa",
        "saccade_duration_rolling__markov",
        "saccade_duration_rolling__arma__*",
        "saccade_duration_rolling__garch",
    ],
]

dimensionality_reduction_names = ["lasso", "PCA"]

new_feature_group_regexes = [
    [
        "^duration_rolling__fft_aggregated__*",
        "^duration_rolling__markov",
        "^duration_rolling__arma__*",
        "^duration_rolling__garch",
    ],
    [
        "saccade_length_rolling__fft_aggregated__*",
        "saccade_length_rolling__markov",
        "saccade_length_rolling__arma__*",
        "saccade_length_rolling__garch",
    ],
    [
        "saccade_duration_rolling__fft_aggregated__*",
        "saccade_duration_rolling__markov",
        "saccade_duration_rolling__arma__*",
        "saccade_duration_rolling__garch",
    ],
    [
        "pupil_diameter_rolling__lhipa",
    ],
]


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
            datasets[dataset_name] = self.handle_nan(dataset)
            labelss[dataset_name] = normalize_series(labels)
        return datasets, labelss

    def run_experiments(self):
        results_list = []
        # dataset_combinations = [
        #    (set(in_study), list(set(self.dataset_names) - set(in_study))[0])
        #    for in_study in itertools.combinations(
        #        self.dataset_names, len(self.dataset_names) - 1
        #    )
        # ]
        dataset_combinations = [
            ([dataset_combination[0]], dataset_combination[1])
            for dataset_combination in itertools.permutations(self.dataset_names, 2)
        ]
        # dataset_combinations = [(self.dataset_names, self.dataset_names[0])]
        print(dataset_combinations)
        for dataset_combination in dataset_combinations:
            for feature_combination in new_feature_group_regexes:
                for dimensionality_reduction_name in dimensionality_reduction_names:
                    in_study_names, oos_name = dataset_combination

                    start = time.time()
                    results = self.run_experiment(
                        in_study_names,
                        oos_name,
                        feature_combination,
                        dimensionality_reduction_name,
                    )
                    end = time.time()
                    print(
                        (end - start),
                        f"An experiment takes this long {dataset_combination}, {feature_combination}",
                    )
                    results["dimensionality_reduction"] = dimensionality_reduction_name
                    results["out_of_study"] = oos_name
                    results["in_study"] = in_study_names
                    results["feature_combinations"] = feature_combination
                    results_list.append(results)
        result_df = pd.DataFrame(results_list)

        globals.comet_logger = CometExistingExperiment(
            api_key=globals.flags.comet_api_key,
            previous_experiment=globals.comet_logger.get_key(),
        )
        globals.comet_logger.log_dataframe_profile(result_df)

    def run_experiment(
        self,
        in_study_names,
        out_of_study_name,
        feature_combination,
        dimensionality_reduction_name,
    ):
        comet_exp = CometExperiment(
            api_key=globals.flags.comet_api_key, project_name="ideal-pancake"
        )

        print("Starting Experiment with")
        print(f"IN STUDY: {in_study_names}")
        print(f"OUT OF STUDY: {out_of_study_name}")
        print(f"FEATURES: {feature_combination}")
        # Run experiment

        dataset, labels = self.merge_datasets(in_study_names)
        dataset = combine_regexes_and_filter_df(dataset, feature_combination)
        oos_dataset = combine_regexes_and_filter_df(
            self.datasets[out_of_study_name], feature_combination
        )
        experiment = Experiment(
            dataset=dataset,
            labels=labels,
            oos_dataset=oos_dataset,
            oos_labels=self.labels[out_of_study_name],
            dimensionality_reduction_name=dimensionality_reduction_name,
            comet_exp=comet_exp,
        )
        metrics, prediction_and_labels = experiment.run_experiment()

        # Logging
        comet_exp.set_name(globals.flags.experiment_name)
        comet_exp.log_metrics(metrics)
        comet_exp.log_other("in-study", in_study_names)
        comet_exp.log_other("out-of-study", out_of_study_name)
        comet_exp.log_other("features", feature_combination)
        return {**metrics, **prediction_and_labels}

    def merge_datasets(self, dataset_combination):
        datasets = [self.datasets[dataset_name] for dataset_name in dataset_combination]
        labels = [self.labels[dataset_name] for dataset_name in dataset_combination]
        return pd.concat(datasets), pd.concat(labels)

    def handle_nan(self, df):
        return df.fillna(0)


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
