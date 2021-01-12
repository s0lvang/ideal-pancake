import json


class DatasetConfig:
    def __init__(self):
        pass

    def __str__(self):
        variables = vars(self).copy()
        ts_fresh = ", ".join(variables.pop("tsfresh_features", {}).keys())
        experimenter = variables.pop("experimenter")
        return f"{json.dumps(variables)} tsfresh_features: ({ts_fresh}) experimenter: ({experimenter.__name__})"
