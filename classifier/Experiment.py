from classifier.utils import log_hyperparameters_to_comet
import numpy as np
from classifier import evaluate, pipelines
from sklearn.model_selection import train_test_split, RandomizedSearchCV


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
        grid_params = self.get_random_grid()
        print(grid_params)
        pipeline = RandomizedSearchCV(
            pipeline,
            grid_params,
            n_iter=2,
            cv=2,
            error_score=np.inf,
            scoring="neg_root_mean_squared_error",
        )
        pipeline.fit(data_train, labels_train)

        print(pipeline.best_score_, "THIS IS THE BEST SCORE")
        print(pipeline.get_params())
        log_hyperparameters_to_comet(pipeline, self.comet_exp)
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
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=30)]
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
        alpha = [1]
        random_forest_grid = {
            # "classifier__n_estimators": n_estimators,
            "classifier__RF__max_depth": max_depth,
            # "classifier__min_samples_split": min_samples_split,
            # "classifier__min_samples_leaf": min_samples_leaf,
            # "classifier__max_features": max_features,
            # "classifier__bootstrap": bootstrap,
        }
        knn_grid = {"classifier__KNN__n_neighbors": [1, 2, 3]}
        SVC_grid = {"classifier__SVR__C": [1, 2, 3]}

        if self.dimensionality_reduction_name == "lasso":
            dim_grid = {"lasso__estimator__alpha": alpha}
        else:
            dim_grid = {"PCA__n_components": alpha}

        return {**random_forest_grid, **knn_grid, **SVC_grid, **dim_grid}

    def __str__(self):
        return super().__str__()
