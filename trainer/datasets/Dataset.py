from trainer.FileRefence import FileReference
import os
from google.cloud import storage
from trainer import globals


class Dataset:
    def __init__(self, name):
        self.name = name

    def data_and_labels(self):
        validate_config()
        file_references = self.get_file_references("data/")
        metadata_references = self.get_file_references("metadata/")
        data, labels = self.prepare_files(file_references, metadata_references)
        return data, labels

    def get_file_references(self, directory_name):
        if globals.FORCE_LOCAL_FILES:
            file_references = get_file_names_from_directory(
                f"datasets/{self.name}/{directory_name}"
            )
        else:
            file_references = get_blobs_from_gcs(
                bucket_name=self.name, prefix=directory_name
            )
        return file_references

    def __str__(self):
        return f"{self.name}"


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


def get_blobs_from_gcs(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_references = list(
        map(FileReference, filter(lambda file: file.name != prefix, blobs))
    )
    return file_references


def validate_config():
    validate_download_settings()


def validate_download_settings():
    if globals.FORCE_LOCAL_FILES and globals.FORCE_GCS_DOWNLOAD:
        raise ValueError(
            "Both force_local_files and force_gcs_download cannot be true at the same time."
        )
