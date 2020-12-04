import pandas as pd
import numpy as np
import cv2
from trainer.datasets import datasets


def prepare_files(file_references, metadata_references, label_column, id_column):
    with metadata_references[0].open("r") as f:
        metadata_file = pd.read_csv(f)

    grouped_files = group_file_references_by_subject_id(file_references)

    subjects_frames = []
    subjects_labels = []
    for (id, group) in grouped_files.items():
        subject_frames, subject_label = prepare_subject(
            id, group, metadata_file, label_column, id_column
        )
        subjects_frames.append(subject_frames)
        subjects_labels.append(subject_label)
    subjects_frames = np.array(subjects_frames)
    subjects_labels = np.array(subjects_labels)
    print("prepare subject returning")
    return subjects_frames, subjects_labels


def prepare_subject(id, file_references, metadata_file, label_column, id_column):
    frames_list = [
        read_and_resize_image(file_reference)
        for file_reference in sorted(file_references)
    ]
    frames = np.array(frames_list)
    label = heatmap_label(
        metadata_file, id, subject_column=id_column, score_column=label_column
    )
    return frames, label


def read_and_resize_image(file_reference):
    with file_reference.open("rb") as f:
        file_content = f.read()

    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
    return cv2.resize(image, (100, 100))


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
