from classifier.datasets.Timeseries import Timeseries


class CSCW(Timeseries):
    def __init__(self):
        super().__init__("cscw")
        self.label = "Posttest.Score"
        self.labels_are_categorical = True
        self.encoding = {}

    def __str__(self):
        return super().__str__()
