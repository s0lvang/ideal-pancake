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
    print(timeseries[0], "arma", "beginning")
    arma_coeff_names = ["exog", "ar", "ma"]
    best_arma_coeffs = optimize_arma(timeseries, param)
    print(timeseries[0], "arma", "end", best_arma_coeffs)
    return [(name, coeff) for name, coeff in zip(arma_coeff_names, best_arma_coeffs)]


@set_property("fctype", "combiner")
def garch(d, param):
    timeseries = copy.deepcopy(d)
    print(timeseries[0], "garch", "beginning")
    garch_coeff_names = ["mu", "omega", "alpha", "gamma", "beta"]
    best_garch_coeffs = optimize_garch(timeseries, param)
    print(timeseries[0], "garch", "end", best_garch_coeffs)
    return [(name, coeff) for name, coeff in zip(garch_coeff_names, best_garch_coeffs)]


@set_property("fctype", "simple")
def lhipa(d):
    timeseries = copy.deepcopy(d)
    print(timeseries[0], "lhipa", "beginning")
    lhipa_value = np.nan
    try:
        lhipa_value = calculate_lhipa(timeseries)
    except Exception as e:
        print(e)
    finally:
        print(timeseries[0], "lhipa", "end", lhipa_value)
        return lhipa_value


@set_property("fctype", "combiner")
def markov(d, param):
    timeseries = copy.deepcopy(d)
    print(timeseries[0], "markov", "beginning")
    transition_matrix = calculate_markov(timeseries, param)
    print(timeseries[0], "markov", "end")
    return [(str(index), value) for index, value in enumerate(transition_matrix)]


def load_custom_functions():
    custom_functions = [garch, lhipa, arma, markov]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)