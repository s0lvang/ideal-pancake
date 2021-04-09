from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class WeightedAverageEnsemble(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        super().fit(X_train, y_train, sample_weight=sample_weight)
        self.weights = [
            self.score_estimator(X_test, y_test, estimator)
            for estimator in self.estimators_
        ]
        print(self.weights, "THIS IS THE WEIGTHS")
        return self

    def score_estimator(self, X, y, estimator):
        prediction = estimator.predict(X)
        return 1 - mean_squared_error(y, prediction, squared=False)
