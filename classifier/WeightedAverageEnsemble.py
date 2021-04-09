from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split


class WeightedAverageEnsemble(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        super().fit(X_train, y_train, sample_weight=sample_weight)
        # self.weights = [
        #    estimator.score(X_test, y_test) for estimator in self.estimators_
        # ]
        print(self.weights, "THIS IS THE WEIGTHS")
        return self
