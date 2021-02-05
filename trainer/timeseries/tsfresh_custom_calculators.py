from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
import math
import numpy as np
import pywt


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
    # find max decomposition level
    w = pywt.Wavelet("sym16")
    maxlevel = pywt.dwt_max_level(len(d), filter_len=w.dec_len)
    # set high and low frequency band indeces
    hif, lof = 1, int(maxlevel / 2)
    # get detail coefficients of pupil diameter signal d
    cD_H = pywt.downcoef("d", d, "sym16", "per", level=hif)
    cD_L = pywt.downcoef("d", d, "sym16", "per", level=lof)
    # normalize by 1/ 2j
    cD_H[:] = [x / math.sqrt(2 ** hif) for x in cD_H]
    cD_L[:] = [x / math.sqrt(2 ** lof) for x in cD_L]
    # obtain the LH:HF ratio
    cD_LH = cD_L
    for i in range(len(cD_L)):
        cD_LH[i] = cD_L[i] / cD_H[int((2 ** lof) / (2 ** hif)) * i]
    # detect modulus maxima , see Duchowski et al. [15]
    cD_LHm = modmax(cD_LH)
    # threshold using universal threshold λuniv = σˆ (2logn)
    # where σˆ is the standard deviation of the noise
    lambda_univ = np.std(cD_LHm) * math.sqrt(2.0 * np.log2(len(cD_LHm)))
    cD_LHt = pywt.threshold(cD_LHm, lambda_univ, mode="less")
    # get signal duration (in seconds)
    print(d[-1])
    sampling_rate = 250
    tt = len(d) // sampling_rate
    # compute LHIPA
    ctr = 0
    LHIPA = -1337
    for i in range(len(cD_LHt)):
        if math.fabs(cD_LHt[i]) > 0:
            ctr += 1
            LHIPA = float(ctr) / tt
    return LHIPA


def modmax(d):
    # compute signal modulus
    m = [0.0] * len(d)
    for i in range(len(d)):
        m[i] = math.fabs(d[i])
    # if value is larger than both neighbours , and strictly
    # larger than either , then it is a local maximum
    t = [0.0] * len(d)
    for i in range(len(d)):
        ll = m[i - 1] if i >= 1 else m[i]
        oo = m[i]
        rr = m[i + 1] if i < len(d) - 2 else m[i]
        if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
            # compute magnitude
            t[i] = math.sqrt(d[i] ** 2)
        else:
            t[i] = 0.0
    return t


def load_custom_functions():
    custom_functions = [garch, lhipa]
    for func in custom_functions:
        setattr(feature_calculators, func.__name__, func)