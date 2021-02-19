from sklearn.model_selection import RandomizedSearchCV
from classifier import model
from classifier.datasets.Dataset import Dataset

from classifier.Labels import Labels
from classifier import globals
from scipy.stats import uniform

import pandas as pd
import cv2
import numpy as np


class Heatmap(Dataset):
    def __init__(self, name):
        super().__init__(name)
        self.image_size = (150, 100)

    def get_features_from_gcs(self):
        data, labels = globals.dataset.download_premade_features()
        labels = Labels(labels, globals.dataset.labels_are_categorical)
        oos_data, oos_labels = globals.out_of_study_dataset.download_premade_features()
        oos_labels = Labels(
            oos_labels, globals.out_of_study_dataset.labels_are_categorical
        )
        return (
            data,
            labels,
            oos_data,
            oos_labels,
        )

    def run_experiment(self, flags):
        (
            data,
            labels,
            oos_data,
            oos_labels,
        ) = self.get_features_from_gcs()

        (data_train, data_test) = labels.train_test_split(data)
        pipeline = model.build_pipeline()

        grid_params = self.get_random_grid()
        pipeline = RandomizedSearchCV(pipeline, grid_params, n_iter=1, cv=3)
        pipeline.fit(data_train, labels.train)

        print("Best Score: ", pipeline.best_score_)
        print("Best Params: ", pipeline.best_params_)
        best_pipeline = pipeline.best_estimator_
        scores = model.evaluate_model(
            best_pipeline,
            data_test,
            labels,
            oos_data,
            oos_labels,
        )

        return scores

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
        alphas = uniform()
        random_grid = {
            "Lasso__estimator__alpha": alphas,
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

    def group_file_references_by_subject_id(self, file_references):
        grouped = {}

        for file_reference in file_references:
            id = self.subject_id(file_reference)
            grouped[id] = [*grouped.get(id, []), file_reference]

        return grouped
