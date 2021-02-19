from classifier.datasets.Timeseries import Timeseries


class EMIP(Timeseries):
    def __init__(self):
        super().__init__("emip-enhanced")
        self.label = "expertise_programming"
        self.labels_are_categorical = True

    def __str__(self):
        return super().__str__()
