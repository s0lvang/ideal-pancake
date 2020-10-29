import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack
import itertools
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from arch import arch_model

data = pd.read_csv("./EEG-dummy-data.csv")
samples = 25


def moving_average(timeseries, sample_size):
    return timeseries.rolling(window=sample_size).mean()


def fft(signals_in_time_domain):
    signals_numpy = signals_in_time_domain.to_numpy()
    return sp.fftpack.fft(signals_numpy)


def inverse_fft(signals_in_frequency_domain):
    return sp.fftpack.ifft(signals_in_frequency_domain)


def pdq_combinations(ceiling):
    p = d = q = range(0, ceiling)
    return list(itertools.product(p, d, q))


# Should take training data, after splitting
def arima(timeseries, pdq):
    arima_model = ARIMA(timeseries, order=pdq)
    fit = arima_model.fit()
    aic = fit.aic
    return fit, aic


def optimize_arima(timeseries, param_ceiling):
    params = pdq_combinations(param_ceiling)
    best_parameter_set = (0, 0, 0)
    best_aic = math.inf

    for parameter_set in params:
        try:
            fit, aic = arima(timeseries, parameter_set)

            if aic < best_aic:
                best_aic = aic
                best_parameter_set = parameter_set
                print(parameter_set, aic, "new best fit")

            else:
                print(parameter_set, aic)
        except Exception as e:
            print(e)
            continue
    print(best_parameter_set, best_aic, "Completed, best fit")
    return best_parameter_set


def garch(timeseries, poq):
    model = arch_model(timeseries, vol="garch", p=poq[0], o=poq[1], q=poq[2])
    fit = garch.fit()
    return fit
