from trainer.FileRefence import FileReference
from collections import Counter
from random import uniform
from sklearn import model_selection


class Labels:
    def __init__(self, labels, categorical):
        self.original_labels = labels
        self.original_max = max(self.original_labels)
        self.original_min = min(self.original_labels)

        if categorical:
            self.labels = labels
        else:
            self.labels = self.k_means_cluster(labels)

        self.labels = self.normalized_labels()
        self.clusters = self.create_clusters()
        self.labels = self.labels_as_values_in_cluster()
        self.max = max(self.labels)
        self.min = min(self.labels)

        # Fallbacks for the case where we don't call train_test_split
        self.train = self.labels
        self.test = self.labels
        self.original_labels_train = self.original_labels
        self.original_labels_test = self.original_labels

    def train_test_split(self, data):
        (
            data_train,
            data_test,
            labels_train_indicies,
            labels_test_indicies,
        ) = model_selection.train_test_split(data, self.labels.index)

        self.train = self.labels[labels_train_indicies]
        self.test = self.labels[labels_test_indicies]
        self.original_labels_train = self.original_labels[labels_train_indicies]
        self.original_labels_test = self.original_labels[labels_test_indicies]
        return (data_train, data_test)

    def create_clusters(self):
        clusters = {}
        sums = 0
        c = Counter(self.labels)

        percentages = {key: value / len(self.labels) for (key, value) in c.items()}
        for (key, value) in sorted(percentages.items()):
            clusters[key] = (sums, sums + value)
            sums += value
        return clusters

    def labels_as_values_in_cluster(self):
        return self.labels.apply(lambda label: uniform(*self.clusters[label]))

    def k_means_cluster(self, labels):
        return labels

    def normalized_labels(self):
        return (self.labels - self.labels.min()) / (
            self.labels.max() - self.labels.min()
        )

    def get_cluster_from_value(self, value):
        for key, cluster in self.clusters.items():
            if value > cluster[0] and value < cluster[1]:
                return key
        raise Exception("not in any cluster in ", self.clusters)

    def get_clusters_from_values(self, values):
        return [self.get_cluster_from_value(value) for value in values]

    def __str__(self):
        return self.original_labels_test.__str__()