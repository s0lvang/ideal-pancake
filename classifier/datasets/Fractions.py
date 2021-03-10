from classifier.datasets.Dataset import Dataset


class Fractions(Dataset):
    def __init__(self):
        super().__init__("fractions")
        self.label = "Post_SumOfCorrect_NewSum"
        self.labels_are_categorical = True
        self.encoding = {}

    def __str__(self):
        return super().__str__()
