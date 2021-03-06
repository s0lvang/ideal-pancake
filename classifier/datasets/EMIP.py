from classifier.datasets.Dataset import Dataset


class EMIP(Dataset):
    def __init__(self):
        super().__init__("emip-fixations")
        self.label = "expertise_programming"
        self.labels_are_categorical = True
        self.encoding = {"high": 3, "medium": 2, "low": 1, "none": 0}

    def get_labels(self, label_dataframe):
        return label_dataframe["correct_vehicle"] + label_dataframe["correct_rectangle"]

    def __str__(self):
        return super().__str__()
