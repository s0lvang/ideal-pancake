import itertools
import math
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import sys
import os


def poq_combinations(ceiling):
    p = o = q = range(1, ceiling)
    return list(itertools.product(p, o, q))


def pq_combinations(ceiling):
    p = q = range(1, ceiling)
    return list(itertools.product(p, q))


# Should take training data, after splitting
def fit_arma(timeseries, pq):
    p, q = pq
    order = (p, 0, q)
    arma_model = ARIMA(timeseries, order=order)
    fit = arma_model.fit()
    aic = fit.aic
    return fit, aic


def fit_garch(timeseries, poq):
    garch = arch_model(timeseries, vol="garch", p=poq[0], o=poq[1], q=poq[2])
    fit = garch.fit()
    aic = fit.aic
    return fit, aic


def optimize_model(timeseries, combinations, model):
    best_fit = None
    best_aic = math.inf

    for parameter_set in combinations:
        try:
            fit, aic = model(timeseries, parameter_set)
            if aic < best_aic:
                best_aic = aic
                best_fit = fit
        except Exception as e:
            print(e)
            continue
    return best_fit


def optimize_garch(timeseries, param_ceiling):
    sys.stdout = open(os.devnull, "w")
    combinations = poq_combinations(param_ceiling)
    fit = optimize_model(timeseries, combinations, fit_garch)
    sys.stdout = sys.__stdout__
    return fit.params


def optimize_arma(timeseries, param_ceiling):
    combinations = pq_combinations(param_ceiling)
    fit = optimize_model(timeseries, combinations, fit_arma)
    return fit.params
