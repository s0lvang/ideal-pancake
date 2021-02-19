from sklearn.model_selection import RandomizedSearchCV
from feature_generation import model
from feature_generation.datasets.Dataset import Dataset
from feature_generation import globals
from scipy.stats import uniform

import pandas as pd
import cv2
import numpy as np
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
        subjects_labels = subjects_labels
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

    def generate_features(self):
        data, labels = self.data_and_labels()

        preprocessing_pipeline = model.create_vgg_pipeline()
        data = preprocessing_pipeline.fit_transform(data)

        if globals.flags.environment == "remote":
            globals.dataset.upload_features_to_gcs(data, labels)

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
