from google.cloud.storage.blob import Blob
import re
import os
from trainer import globals


class FileReference:
    blob = None
    reference = ""

    def __init__(self, reference_or_blob):
        if isinstance(reference_or_blob, str):
            self.reference = reference_or_blob
        elif isinstance(reference_or_blob, Blob):
            self.blob = reference_or_blob

            self.reference = os.path.join(
                "datasets", self.blob.bucket.name, self.blob.name
            )

    def download_to_filename(self, destination_file_name):
        if not Blob:
            raise ValueError(f"{self.__str__()} does not have a blob set")
        return self.blob.download_to_filename(destination_file_name)

    def open(self, *args):
        if Blob and not globals.FORCE_LOCAL_FILES:
            self.cached_download_data()
        return open(self.reference, *args)

    def cached_download_data(self):
        split_name = self.blob.name.split("/")
        data_directory = os.path.join("datasets", self.blob.bucket.name)
        for i in range(len(split_name) - 1):
            directory = os.path.join(data_directory, "/".join(split_name[: i + 1]))
            if not os.path.exists(directory):
                os.makedirs(directory)
        if not os.path.isfile(self.reference):
            self.download_to_filename(self.reference)

    def human_sorting_keys(self, reference):
        return [
            int(part) if part.isdigit() else part
            for part in re.findall(r"[^0-9]|[0-9]+", reference)
        ]

    def __gt__(self, other):
        return self.human_sorting_keys(self.reference) > self.human_sorting_keys(
            other.reference
        )

    def __str__(self) -> str:
        return self.reference

    def __repr__(self) -> str:
        return self.__str__()
