from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
import numpy as np


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

    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        print(self._weights_not_none, "weights")
        print(np.array(self._predict(X)), "pred")
        return np.average(
            self._predict(X).astype("float"), axis=1, weights=self._weights_not_none
        )
