import json


class DatasetConfig:
    def __init__(self):
        pass

    def __str__(self):
        variables = vars(self).copy()
        ts_fresh = ", ".join(variables.pop("TSFRESH_FEATURES", {}).keys())
        file_preparer = variables.pop("file_preparer")
        experimenter = variables.pop("experimenter")
        return f"{json.dumps(variables)} TS_FRESH_FEATURES: ({ts_fresh}) file_preparer: ({file_preparer.__name__}) experimenter: ({experimenter.__name__})"
