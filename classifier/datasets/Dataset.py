from classifier.utils import dump_object, download_object
import os
import json


class Dataset:
    def __init__(self, name):
        self.name = name
        self.labels_are_categorical = False

    def upload_features_to_gcs(self, features, labels):
        features_output_path = os.path.join(
            "pregenerated-features", self.name, "features"
        )
        dump_object((features, labels), features_output_path)

    def get_labels(self, label_dataframe):
        labels = label_dataframe.loc[:, self.label]
        return labels.apply(lambda x: self.encoding.get(x, x))

    def download_premade_features(self):
        features_path = os.path.join("pregenerated-features", self.name, "features")
        features, labels_df = download_object(features_path)
        return features, self.get_labels(labels_df)

    def __str__(self):
        variables = vars(self).copy()
        ts_fresh = ", ".join(variables.pop("tsfresh_features", {}).keys())
        return f"{json.dumps(variables)} tsfresh_features: ({ts_fresh})"
