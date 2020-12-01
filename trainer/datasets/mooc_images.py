import pandas as pd
import numpy as np
import cv2


def prepare_files(file_references, metadata_references):
    labels = np.array([])
    subjects = np.array([])
    with metadata_references[0].open("r") as f:
        metadata_file = pd.read_csv(f)

    grouped_files = group_file_references_by_subject_id(file_references)

    subjects_frames = []
    subjects_labels = []
    for (id, group) in grouped_files.items():
        subject_frames, subject_label = prepare_subject(id, group, metadata_file)
        subjects_frames.append(subject_frames)
        subjects_labels.append(subject_label)
    subjects_frames = np.array(subjects_frames)
    subjects_labels = np.array(subjects_labels)
    return subjects_frames, subjects_labels


def prepare_subject(id, file_references, metadata_file):
    frames_list = [
        read_and_resize_image(file_reference)
        for file_reference in sorted(file_references)
    ]
    frames = np.array(frames_list)
    label = metadata_file[metadata_file["subject"] == id]["posttest"]
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


def prepare_file(f, metadata_file, dataset, labels):
    subject_id = int(subject_directory)
    subject_directory = os.path.join(directory_name, subject_directory)
    frames_for_subjects = np.array(
        [
            cv2.resize(cv2.imread(os.path.join(subject_directory, file)), (300, 170))
            for file in sorted(
                os.listdir(subject_directory),
                key=lambda var: [
                    int(x) if x.isdigit() else x
                    for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )
        ]
    )
    label = metadata[metadata["subject"] == int(subject_id)]["posttest"]
    # print(subject_id)
    # print(metadata)
    images = (
        np.concatenate((images, np.array([frames_for_subjects])))
        if images.size
        else np.array([frames_for_subjects])
    )
    labels = np.hstack((labels, label))
