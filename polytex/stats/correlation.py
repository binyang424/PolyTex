# Determining Stochastic Characteristics of Tow Loci

import matplotlib.pyplot as plt
import numpy as np


def rmsd(x):
    """
    Root-Mean-Square Deviation (RMSD) of a random variable x:

    sigma = sqrt( 1/N * sum(x^2) )

    """
    return np.sqrt(np.mean(x ** 2))


def pearson_correlation(x, y):
    """
    Pearson's correlation parameter for two variables x and y:

    C = Cov(x, y) / (sigma_x * sigma_y)

    Parameters
    ----------
    x : 1D array_like
        First variable.
    y : 1D array_like
        Second variable. Must have the same shape as x.

    Returns
    -------
    C : float
        Pearson's correlation parameter
    """
    xy = x * y
    C = np.mean(xy) / (rmsd(x) * rmsd(y))
    return C


def sin_theta(theta_min, theta_max, step, plot=False):
    """
    Input the range of theta values, return a list of sin(theta)

    Parameters
    ----------
    theta_min : float
        Minimum value of theta in radians.
    theta_max : float
        Maximum value of theta in radians.
    step : float
        Step size of theta in radians.
    plot : bool, optional
        If True, plot sin(theta) vs theta.

    Returns
    -------
    sin_theta : 1D array_like
        List of sin(theta) values.
    """
    theta = np.arange(theta_min, theta_max, step)

    if plot:
        plt.scatter(theta, np.sin(theta), s=12, c='b', marker='o')
        plt.plot(theta, np.sin(theta), c='r')

        plt.xlabel('$\\theta (\degree)$')
        plt.ylabel('sin($\\theta$)')
        plt.show()

    return theta, np.sin(theta)


def linear(x, a):
    """
    Linear function for fitting correlation vs k

    Parameters
    ----------
    x : 1D array_like
        x values.
    a : float
        Parameter.

    Returns
    -------
    y : 1D array_like
        y values.
    """
    delta = x * 2 * np.pi / 40
    return 1 - delta / a


if __name__ == '__main__':
    """ Make up some data for Pearson's correlation analysis """
    theta_min = 0
    theta_max = 2 * np.pi * 8

    n = 40
    step = 2 * np.pi / n

    theta, x = sin_theta(theta_min, theta_max, step, plot=True)

    """ Calculate Pearson's correlation parameter for all possible k """
    for k in range(5, 15):
        # Create paired data by move x[:k] to the end of x and assign to y
        y = np.concatenate((x[k:], x[:k]))
        C = pearson_correlation(x, y)

        try:
            corr = np.vstack((corr, [k, C]))
        except:
            corr = np.array([k, C])

    """ plot correlation vs k """
    plt.scatter(corr[:, 0], corr[:, 1], s=12, c='b', marker='o')
    plt.plot(corr[:, 0], corr[:, 1], c='r')
    plt.hlines(0, 0, n, colors='k', linestyles='dashed')
    plt.xlabel('k (interval between paired points)')
    plt.ylabel('C (Pearson\'s correlation parameter)')
    plt.show()

    """ Calculation of Correlation length by linear fitting of correlation vs k"""
    from scipy.optimize import curve_fit
    # curve_fit() returns two arrays: popt and pcov
    popt, pcov = curve_fit(linear, corr[:, 0], corr[:, 1])