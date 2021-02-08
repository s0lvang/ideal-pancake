import pandas as pd
import itertools
import math
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


def pdq_combinations(ceiling):
    p = d = q = range(1, ceiling)
    return list(itertools.product(p, d, q))


# Should take training data, after splitting
def arima(timeseries, pdq):
    arima_model = ARIMA(timeseries, order=pdq)
    fit = arima_model.fit()
    aic = fit.aic
    return fit, aic


def optimize_arima(timeseries, param_ceiling):
    params = pdq_combinations(param_ceiling)
    best_parameter_set = None
    best_aic = math.inf

    for parameter_set in params:
        try:
            fit, aic = arima(timeseries, parameter_set)

            if aic < best_aic:
                best_aic = aic
                best_parameter_set = fit.params
                print(parameter_set, aic, "new best fit")
                print(fit.param_terms, aic, "new best fit")
                print(fit.arparams)
                print(fit.maparams)

            else:
                print(parameter_set, aic)
        except Exception as e:
            print(e)
            continue
    print(best_parameter_set, best_aic, "Completed, best fit")
    return best_parameter_set


def garch(timeseries, poq):
    garch = arch_model(timeseries, vol="garch", p=poq[0], o=poq[1], q=poq[2])
    fit = garch.fit()
    return fit