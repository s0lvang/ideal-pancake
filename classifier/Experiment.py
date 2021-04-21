from classifier.utils import log_hyperparameters_to_comet
import numpy as np
from classifier import evaluate, pipelines
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from ray.tune.sklearn import TuneGridSearchCV


class Experiment:
    def __init__(
        self,
        dataset,
        labels,
        oos_dataset,
        oos_labels,
        dimensionality_reduction_name,
        comet_exp,
    ):
        self.dataset = dataset
        self.labels = labels
        self.oos_dataset = oos_dataset
        self.oos_labels = oos_labels
        self.dimensionality_reduction_name = dimensionality_reduction_name
        self.comet_exp = comet_exp

    def run_experiment(self):
        """Testbed for running model training and evaluation."""
        (
            data_train,
            data_test,
            labels_train,
            labels_test,
        ) = train_test_split(self.dataset, self.labels)

        pipeline = pipelines.build_ensemble_regression_pipeline(
            self.dimensionality_reduction_name
        )
        grid = self.get_random_grid()
        tune_search = TuneGridSearchCV(
            pipeline,
            grid,
            max_iters=10,
            scoring="neg_root_mean_squared_error",
            error_score=np.inf,
        )
        tune_search.fit(data_train, labels_train)

        log_hyperparameters_to_comet(pipeline, self.comet_exp)
        best_pipeline = tune_search.best_estimator_

        metrics = evaluate.evaluate_model(
            best_pipeline,
            data_test,
            labels_test,
            self.oos_dataset,
            self.oos_labels,
        )
        return metrics

    def get_random_grid(self):

        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=30)]
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)

        random_forest_grid = {
            "classifier__RF__n_estimators": n_estimators,
            "classifier__RF__max_depth": max_depth,
        }
        knn_grid = {"classifier__KNN__n_neighbors": [3, 5, 7, 9]}
        SVR_grid = {
            "classifier__SVR__C": [1, 3, 6, 9, 11],
            "classifier__SVR__degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "classifier__SVR__gamma": [0.1, 0.5, 1, 1.5, 2, 2.5],
        }
        lasso_grid = {"lasso__estimator__alpha": [0, 0.20, 0.40, 0.60, 0.80, 1]}
        PCA_grid = {"PCA__n_components": [None, 0.20, 0.40, 0.60, 0.80, 0.99]}

        if self.dimensionality_reduction_name == "lasso":
            dim_grid = lasso_grid
        else:
            dim_grid = PCA_grid

        return {**random_forest_grid, **knn_grid, **SVR_grid, **dim_grid}

    def __str__(self):
        return super().__str__()
