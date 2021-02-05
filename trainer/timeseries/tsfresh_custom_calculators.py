from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
import math
from .lhipa import calculate_lhipa


@set_property("fctype", "simple")
def garch(x):
    """
    The description of your feature

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool, int or float
    """
    # Calculation of feature as float, int or bool
    return 42


@set_property("fctype", "simple")
def lhipa(d):
    return calculate_lhipa(d)


def load_custom_functions():
    custom_functions = [garch, lhipa]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)