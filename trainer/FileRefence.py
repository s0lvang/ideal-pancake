from google.cloud.storage.blob import Blob
import re
import os
from trainer import globals


class FileReference:
    reference = ""

    def __init__(self, reference_or_blob):
        self.reference = reference_or_blob

    def open(self, *args):
        return open(self.reference, *args)

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
