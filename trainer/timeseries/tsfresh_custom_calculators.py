from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators


@set_property("fctype", "simple")
def yolo(x):
    """
    The description of your feature

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool, int or float
    """
    # Calculation of feature as float, int or bool
    return 42


def load_custom_functions():
    custom_functions = [yolo]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)