from feature_generation.timeseries.garch_and_arma import optimize_arma, optimize_garch
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
from .lhipa import calculate_lhipa
from .markov import calculate_markov
import numpy as np
import copy


@set_property("fctype", "combiner")
def arma(d, param):
    timeseries = copy.deepcopy(d)
    arma_coeff_names = ["exog", "ar", "ma"]
    best_arma_coeffs = optimize_arma(timeseries, param)
    return [(name, coeff) for name, coeff in zip(arma_coeff_names, best_arma_coeffs)]


@set_property("fctype", "combiner")
def garch(d, param):
    timeseries = copy.deepcopy(d)
    garch_coeff_names = ["mu", "omega", "alpha", "gamma", "beta"]
    best_garch_coeffs = optimize_garch(timeseries, param)
    return [(name, coeff) for name, coeff in zip(garch_coeff_names, best_garch_coeffs)]


@set_property("fctype", "simple")
def lhipa(d):
    timeseries = copy.deepcopy(d)
    lhipa_value = np.nan
    try:
        lhipa_value = calculate_lhipa(timeseries)
    except Exception as e:
        print(e)
    finally:
        return lhipa_value


@set_property("fctype", "combiner")
def markov(d, param):
    timeseries = copy.deepcopy(d)
    transition_matrix = calculate_markov(timeseries, param)
    return [(str(index), value) for index, value in enumerate(transition_matrix)]


def load_custom_functions():
    custom_functions = [garch, lhipa, arma, markov]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)