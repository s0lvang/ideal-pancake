from classifier.utils import log_hyperparameters_to_comet
import numpy as np
from classifier import evaluate, pipelines
from sklearn.model_selection import train_test_split


class Experiment:
    def __init__(self, dataset, labels, oos_dataset, oos_labels):
        self.dataset = dataset
        self.labels = labels
        self.oos_dataset = oos_dataset
        self.oos_labels = oos_labels

    def run_experiment(self):
        """Testbed for running model training and evaluation."""
        (
            data_train,
            data_test,
            labels_train,
            labels_test,
        ) = train_test_split(self.dataset, self.labels)

        pipeline = pipelines.build_ensemble_classification_pipeline()

        # grid_params = self.get_random_grid()
        # pipeline = RandomizedSearchCV(pipeline, grid_params, n_iter=2, cv=2)
        pipeline.fit(data_train, labels_train)

        # log_hyperparameters_to_comet(pipeline)
        best_pipeline = pipeline  # .best_estimator_

        metrics = evaluate.evaluate_model(
            best_pipeline,
            data_test,
            labels_test,
            self.oos_dataset,
            self.oos_labels,
        )
        return metrics

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
