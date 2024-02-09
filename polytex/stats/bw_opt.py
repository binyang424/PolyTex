import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def bw_scott(sigma, n=''):
    """
    Scott's rule for bandwidth selection.
    :param sigma: standard deviation of the data, type: float
    :param n: number of data points, type: int
    :return: bandwidth, type: float
    """
    return sigma * (4.0 / 3.0 / n) ** 0.2


def opt_bandwidth(variable, x_test, bw):
    """
    Find the optimal bandwidth by tuning of the `bandwidth` parameter
    via cross-validation and returns the parameter value that maximizes
    the log-likelihood of data.

    Parameters
    ----------
    variable : Numpy array
        A N x 1 dimension numpy array. The data to apply the kernel density estimation.
    x_test : Numpy array
        Test data to get the density distribution.
    bw : list of float
        The bandwidth of the kernels to be tested.
    """

    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, {'bandwidth': bw})
    grid.fit(variable)

    kde = grid.best_estimator_
    log_dens = kde.score_samples(x_test)
    print("the log-likelihood of data: ", log_dens)
    if len(bw) > 1:
        print("optimal bandwidth: " + "{:.4f}".format(kde.bandwidth))

    return kde.bandwidth


def log_likelihood(pdf):
    """
    Calculate the likelihood of the given probability density function.
    The likelihood is:
       ``L = \frac{1}{N}\sum_{i=1}^{N} f(x_i)``
    Parameters
    ----------
    pdf : Numpy array
        The probability density function.

    Returns
    -------
    LL : float
        The log-likelihood of the given probability density function.
    """
    LL = np.average(np.log(pdf))
    return LL