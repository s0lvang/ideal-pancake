from classifier.Labels import Labels
from classifier.utils import log_dataframe_to_comet, log_hyperparameters_to_comet
import numpy as np

from classifier.datasets.Dataset import Dataset
from classifier import model
from classifier import globals


class Timeseries(Dataset):
    def __init__(self, name):
        super().__init__(name)

    def get_features_from_gcs(self):
        data, labels = globals.dataset.download_premade_features()
        labels = Labels(
            labels,
            globals.dataset.labels_are_categorical,
        )
        oos_data, oos_labels = globals.out_of_study_dataset.download_premade_features()
        oos_labels = Labels(
            oos_labels,
            globals.out_of_study_dataset.labels_are_categorical,
        )
        return (
            data,
            labels,
            oos_data,
            oos_labels,
        )

    def run_experiment(self, flags):
        """Testbed for running model training and evaluation."""
        (
            data,
            labels,
            oos_data,
            oos_labels,
        ) = self.get_features_from_gcs()

        (
            data_train,
            data_test,
        ) = labels.train_test_split(data)

        pipeline = model.build_pipeline()

        # grid_params = self.get_random_grid()
        # pipeline = RandomizedSearchCV(pipeline, grid_params, n_iter=2, cv=2)
        pipeline.fit(data_train, labels.train)

        # log_hyperparameters_to_comet(pipeline)
        best_pipeline = pipeline  # .best_estimator_

        scores = model.evaluate_model(
            best_pipeline,
            data_test,
            labels,
            oos_data,
            oos_labels,
        )

    def get_random_grid(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ["auto", "sqrt"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        # Create the random grid
        random_grid = {
            "classifier__n_estimators": n_estimators,
            "classifier__max_depth": max_depth,
            "classifier__min_samples_split": min_samples_split,
            "classifier__min_samples_leaf": min_samples_leaf,
            "classifier__max_features": max_features,
            "classifier__bootstrap": bootstrap,
        }
        return random_grid

    def __str__(self):
        return super().__str__()
