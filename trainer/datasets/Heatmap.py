from trainer.utils import normalize_and_numericalize
from trainer import model
from trainer.datasets.Dataset import Dataset
from trainer import globals

import pandas as pd
import cv2
import numpy as np
from sklearn import model_selection


class Heatmap(Dataset):
    def __init__(self, name):
        super().__init__(name)
        self.image_size = (150, 100)

    def prepare_files(self, file_references, metadata_references):
        label_column = self.label
        id_column = self.subject_id_column

        with metadata_references[0].open("r") as f:
            metadata_file = pd.read_csv(f)
            print(metadata_file)

        grouped_files = group_file_references_by_subject_id(file_references)

        subjects_frames = []
        subjects_labels = []
        for (id, group) in grouped_files.items():
            subject_frames, subject_label = self.prepare_subject(
                id, group, metadata_file, label_column, id_column
            )
            subjects_frames.append(subject_frames)
            subjects_labels.append(subject_label)
        subjects_frames = np.array(subjects_frames)
        subjects_labels = pd.Series(subjects_labels)
        return subjects_frames, subjects_labels

    def prepare_subject(
        self, id, file_references, metadata_file, label_column, id_column
    ):
        frames_list = [
            self.read_and_resize_image(file_reference)
            for file_reference in sorted(file_references)
        ]
        frames = np.array(frames_list)
        label = heatmap_label(
            metadata_file, id, subject_column=id_column, score_column=label_column
        )
        return frames, int(label)

    def read_and_resize_image(self, file_reference):
        with file_reference.open("rb") as f:
            file_content = f.read()

        nparr = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        return cv2.resize(image, self.image_size)

    def prepare_dataset(self, data, labels):
        labels = normalize_and_numericalize(labels)
        return data, labels

    def prepare_datasets(self):
        data, labels = self.data_and_labels()
        data, labels = self.prepare_dataset(data, labels)

        oos_data, oos_labels = globals.out_of_study_dataset.data_and_labels()
        oos_data, oos_labels = self.prepare_dataset(oos_data, oos_labels)

        return data, labels, oos_data, oos_labels

    def run_experiment(self, flags):
        (
            data,
            labels,
            oos_data,
            oos_labels,
        ) = self.prepare_datasets()

        (
            data_train,
            data_test,
            labels_train,
            labels_test,
        ) = model_selection.train_test_split(data, labels)

        pipeline = model.build_lasso_pipeline()

        pipeline.fit(data_train, labels_train)

        scores = model.evaluate_model(
            pipeline, data_test, labels_test, oos_data, oos_labels
        )
        model.store_model_and_metrics(pipeline, scores, flags.job_dir)

        return scores

    def __str__(self):
        return super().__str__()


def group_file_references_by_subject_id(file_references):
    grouped = {}

    for file_reference in file_references:
        id = subject_id(file_reference)
        grouped[id] = [*grouped.get(id, []), file_reference]

    return grouped


def subject_id(file_reference):
    return int(file_reference.reference.split("/")[-2])


def heatmap_label(metadata_file, id, subject_column, score_column):
    return metadata_file[metadata_file[subject_column] == id][score_column]
