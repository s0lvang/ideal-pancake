import pandas as pd
import itertools
import math
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import sys
import os


def pdq_combinations(ceiling):
    p = d = q = range(1, ceiling)
    return list(itertools.product(p, d, q))


# Should take training data, after splitting
def fit_arima(timeseries, pdq):
    arima_model = ARIMA(timeseries, order=pdq)
    fit = arima_model.fit()
    aic = fit.aic
    return fit, aic


def fit_garch(timeseries, poq):
    garch = arch_model(timeseries, vol="garch", p=poq[0], o=poq[1], q=poq[2])
    fit = garch.fit()
    aic = fit.aic
    return fit, aic


def optimize_model(timeseries, param_ceiling, model):
    params = pdq_combinations(param_ceiling)
    best_fit = None
    best_aic = math.inf

    for parameter_set in params:
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
    fit = optimize_model(timeseries, param_ceiling, fit_garch)
    sys.stdout = sys.__stdout__
    return fit.params


def optimize_arima(timeseries, param_ceiling):
    fit = optimize_model(timeseries, param_ceiling, fit_arima)
    return fit.params
