from feature_generation.timeseries.garch_and_arma import optimize_arma, optimize_garch
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
from .lhipa import calculate_lhipa
from .markov import calculate_markov


@set_property("fctype", "combiner")
def arma(d, param):
    arma_coeff_names = ["exog", "ar", "ma"]
    best_arma_coeffs = optimize_arma(d, param)

    return [(name, coeff) for name, coeff in zip(arma_coeff_names, best_arma_coeffs)]


@set_property("fctype", "combiner")
def garch(d, param):
    garch_coeff_names = ["mu", "omega", "alpha", "gamma", "beta"]
    best_garch_coeffs = optimize_garch(d, param)
    return [(name, coeff) for name, coeff in zip(garch_coeff_names, best_garch_coeffs)]


@set_property("fctype", "simple")
def lhipa(d):
    return calculate_lhipa(d)


@set_property("fctype", "combiner")
def markov(d, param):
    return [
        (str(index), value) for index, value in enumerate(calculate_markov(d, param))
    ]


def load_custom_functions():
    custom_functions = [garch, lhipa, arma, markov]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)