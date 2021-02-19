from feature_generation.FileRefence import FileReference
from feature_generation.utils import dump_object

import os
import json


class Dataset:
    def __init__(self, name):
        self.name = name

    def data_and_labels(self):
        file_references = self.get_file_references("data/")
        metadata_references = self.get_file_references("metadata/")
        data, labels = self.prepare_files(file_references, metadata_references)

        return data, labels

    def get_file_references(self, directory_name):
        file_references = get_file_names_from_directory(
            f"datasets/{self.name}/{directory_name}"
        )
        return file_references

    def upload_features_to_gcs(self, features, labels):
        features_output_path = os.path.join(
            "pregenerated-features", self.name, "features"
        )
        dump_object((features, labels), features_output_path)

    def __str__(self):
        variables = vars(self).copy()
        ts_fresh = ", ".join(variables.pop("tsfresh_features", {}).keys())
        return f"{json.dumps(variables)} tsfresh_features: ({ts_fresh})"


def get_file_names_from_directory(directory_name):
    return recursive_file_names_from_dir(directory_name, [])


def recursive_file_names_from_dir(path, paths):
    if os.path.isdir(path):
        for sub_path in os.listdir(path):
            recursive_file_names_from_dir(os.path.join(path, sub_path), paths)
        return paths
    elif os.path.isfile(path):
        paths.append(FileReference(path))
        return paths
    else:
        raise ValueError(f"Got a path that isn't dir or file: {path}")
