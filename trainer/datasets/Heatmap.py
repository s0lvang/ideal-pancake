from trainer.utils import normalize_and_numericalize
from sklearn.model_selection import RandomizedSearchCV
from trainer import model
from trainer.datasets.Dataset import Dataset
from trainer.Labels import Labels
from trainer import globals
from scipy.stats import uniform

import pandas as pd
import cv2
import numpy as np
from sklearn import model_selection
from abc import ABCMeta, abstractmethod


class Heatmap(Dataset, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__(name)
        self.image_size = (150, 100)

    def prepare_files(self, file_references, metadata_references):
        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f)

        grouped_files = self.group_file_references_by_subject_id(file_references)

        subjects_frames = []
        subjects_labels = []
        for (id, group) in grouped_files.items():
            subject_frames, subject_label = self.prepare_subject(
                id, group, metadata_file
            )
            subjects_frames.append(subject_frames)
            subjects_labels.append(subject_label)
        subjects_frames = np.array(subjects_frames)
        subjects_labels = pd.Series(subjects_labels)
        subjects_labels = Labels(subjects_labels, self.labels_are_categorical)
        return subjects_frames, subjects_labels

    def prepare_subject(self, id, file_references, metadata_file):
        frames_list = [
            self.read_and_resize_image(file_reference)
            for file_reference in sorted(file_references)
        ]
        frames = np.array(frames_list)
        label = self.heatmap_label(
            metadata_file,
            id,
        )
        return frames, label

    @abstractmethod
    def heatmap_label(self, metadata_file, id):
        return NotImplementedError(
            "This is an abstract method it needs to be given in the child class"
        )

    def read_and_resize_image(self, file_reference):
        try:
            with file_reference.open("rb") as f:
                file_content = f.read()
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(
                nparr, cv2.IMREAD_COLOR
            )  # cv2.IMREAD_COLOR in OpenCV 3.1
            return cv2.resize(image, self.image_size)
        except Exception as error:
            print(file_reference)
            raise error

    def prepare_datasets(self):
        data, labels = self.data_and_labels()
        oos_data, oos_labels = globals.out_of_study_dataset.data_and_labels()

        return data, labels, oos_data, oos_labels

    def run_experiment(self, flags):
        (
            data,
            labels,
            oos_data,
            oos_labels,
        ) = self.prepare_datasets()

        (data_train, data_test) = labels.train_test_split(data)

        pipeline = model.build_lasso_pipeline()

        grid_params = self.get_random_grid()
        pipeline = RandomizedSearchCV(pipeline, grid_params, n_iter=1, cv=2)
        pipeline.fit(data_train, labels.train)

        print(pipeline.get_params())
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
        model.store_model_and_metrics(pipeline, scores, flags.job_dir)

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
        bootstrap = [True, False]
        # Create the random grid
        alphas = uniform()
        random_grid = {
            "Lasso__estimator__alpha": alphas,
            "classifier__n_estimators": n_estimators,
            "classifier__max_features": max_features,
            "classifier__max_depth": max_depth,
            "classifier__min_samples_split": min_samples_split,
            "classifier__min_samples_leaf": min_samples_leaf,
            "classifier__bootstrap": bootstrap,
        }
        return random_grid

    def __str__(self):
        return super().__str__()

    @abstractmethod
    def subject_id(self, file_reference):
        raise NotImplementedError(
            "This is an abstract method it needs to be given in the child class"
        )

    def group_file_references_by_subject_id(self, file_references):
        grouped = {}

        for file_reference in file_references:
            id = self.subject_id(file_reference)
            grouped[id] = [*grouped.get(id, []), file_reference]

        return grouped
