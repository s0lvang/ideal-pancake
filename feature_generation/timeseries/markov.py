from hmmlearn import hmm
import numpy as np
import math


def fit_markov(timeseries, n_components):
    hmmmodel = hmm.GaussianHMM(
        n_components=n_components, covariance_type="full", n_iter=100
    )
    hmmmodel.fit(timeseries.reshape(-1, 1))
    # print(hmmmodel.transmat_)
    log_likelyhood = hmmmodel.score(timeseries.reshape(-1, 1))
    aic = -2 * log_likelyhood + 2 * n_components
    return hmmmodel.transmat_, aic


def optimize_markov(timeseries, param_ceiling):
    best_transistion_matrix = None
    best_aic = math.inf

    for i in range(5, param_ceiling):
        try:
            transition_matrix, aic = fit_markov(timeseries, i)
            if aic < best_aic:
                best_aic = aic
                best_transistion_matrix = transition_matrix
        except Exception as e:
            print(e)
            continue
    return best_transistion_matrix


def calculate_markov(timeseries, n_components_ceiling):
    normalized_timeseries = (timeseries - min(timeseries)) / (
        max(timeseries) - min(timeseries)
    )
    bins = [i / 100 for i in range(0, 100, 5)]
    discrete_timeseries = np.digitize(normalized_timeseries, bins)
    transition_matrix = optimize_markov(discrete_timeseries, n_components_ceiling)
    if transition_matrix is None:
        return [np.nan for i in range(n_components_ceiling ** 2)]
    padded_trans_matrix = np.pad(
        transition_matrix.flatten(),
        (0, n_components_ceiling ** 2 - len(transition_matrix)),
    )
    return padded_trans_matrix
