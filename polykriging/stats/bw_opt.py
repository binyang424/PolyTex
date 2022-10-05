import numpy as np


def bw_scott(sigma, n):
    """
    Scott's rule for bandwidth selection.
    :param sigma: standard deviation of the data, type: float
    :param n: number of data points, type: int
    :return: bandwidth, type: float
    """
    return sigma * (4.0 / 3.0 / n) ** 0.2


def mle_bw(f):
    """
    Maximum likelihood estimation of bandwidth.
    :param f: the density function, type: function
    :return: the likelihood of given bandwidth, type: float
    """
    return np.sum(np.log10(f)) / len(np.log10(f).flatten())


def optBandwidth(variable, x_test, bw):
    """
    # minimize the negative log-likelihood
    :param variable:
    :param x_test:
    :param bw:
    :return:
    """
    from scipy.optimize import minimize
    pass
