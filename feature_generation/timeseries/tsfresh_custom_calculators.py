from feature_generation.timeseries.garch_and_arima import optimize_arima, optimize_garch
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
from .lhipa import calculate_lhipa
from .markov import calculate_markov


@set_property("fctype", "combiner")
def arima(d, param):
    arima_coeff_names = ["exog", "ar", "ma"]
    best_arima_coeffs = optimize_arima(d, 3)

    return [(name, coeff) for name, coeff in zip(arima_coeff_names, best_arima_coeffs)]


@set_property("fctype", "combiner")
def garch(d, param):
    garch_coeff_names = ["mu", "omega", "alpha", "gamma", "beta"]
    best_garch_coeffs = optimize_garch(d, 2)
    return [(name, coeff) for name, coeff in zip(garch_coeff_names, best_garch_coeffs)]


@set_property("fctype", "simple")
def lhipa(d):
    return calculate_lhipa(d)


@set_property("fctype", "combiner")
def markov(d, param):
    return [(str(index), value) for index, value in enumerate(calculate_markov(d, 10))]


def load_custom_functions():
    custom_functions = [garch, lhipa, arima, markov]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)