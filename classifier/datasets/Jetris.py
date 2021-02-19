from classifier.datasets.Timeseries import Timeseries


class Jetris(Timeseries):
    def __init__(self):
        super().__init__("jetris")
        self.labels_are_categorical = False

    def __str__(self):
        return super().__str__()