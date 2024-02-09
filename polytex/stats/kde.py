import numpy as np
import scipy
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from .bw_opt import opt_bandwidth
from ..thirdparty.bcolors import bcolors


def kdeScreen(variable, x_test, bw, kernels='gaussian', plot="False"):
    """
    This function estimates the probability density distribution of the input variable
    with the non-parametric kernel density estimation (KDE) method. The local maxima
    and minima of the probability density distribution are identified to decompose the
    input variable into a set of clusters. The former is used as the cluster centers
    and the latter is used as the cluster boundaries.

    Parameters
    ----------
    variable : Numpy array
        A N x 1 dimension numpy array to apply the kernel density estimation.
    x_test : Numpy array
        Test data to get the density distribution. It has the same shape as
        the given variable. It should cover the whole range of the variable.
    bw : float
        The bandwidth of the kernel.
    kernel : string, optional
        The kernel to use. The default is 'gaussian'. The possible values are
        {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}.
    plot : bool, optional
        Whether plot the probability density distribution. The default is False.

    Returns
    -------
    clusters : dictionary
        The index of the cluster centers, cluster boundary and the
        probability density distribution (pdf).

    """
    # check if the variable is 1D array

    if variable.ndim == 1:
        variable = variable.reshape(-1, 1)
    else:
        raise ValueError("The input variable should be a 1D array.")

    model = KernelDensity(kernel=kernels, bandwidth=bw)
    model.fit(variable)

    log_dens = model.score_samples(x_test.reshape(-1, 1))
    pdf_input = np.exp(model.score_samples(variable))

    kde = plt.plot(x_test, np.exp(log_dens), c='cyan')
    _, pdf = kde[0].get_data()
    if plot != "False":
        # x label
        plt.xlabel('Normalized distance (radial)')
        # y label
        plt.ylabel('Probability density')
        plt.show()

    # mask for the local maxima of density
    # argrelextrema: identify the relative extrema of `data`.
    cluster_bounds = np.insert(argrelextrema(pdf, np.less)[0], 0, 0)
    cluster_bounds = np.append(cluster_bounds, -1)
    clusters = {"cluster centers": argrelextrema(pdf, np.greater)[0],  # Indices of local maxima
                "cluster boundary": cluster_bounds,
                "t test": x_test, "pdf": pdf,
                "t input": variable, "pdf input": pdf_input, }  # pdf_input is the pdf of the input variable
    return clusters


def movingKDE(dataset, bw=0.002, windows=1, n_clusters=20, x_test=None):
    """
    This function applies the kernel density estimation (KDE) method to the input
    dataset with a moving window. Namely, the dataset is divided into a set of
    windows and the KDE method is applied to each window. This allows to capture
    more details of geometry changes of a fiber tow.

    Parameters
    ----------
    dataset : Numpy array
        A N x 2 dimension numpy array for kernel density estimation.
        The first colum should be the variable under analysis, the second
        is the label of cross-sections that the variable belongs to.
    bw : Numpy array or float, optional
        A range of bandwidth values for kde operation usually generated with np.arange().
        The optimal bandwidth will be identified within this range and be used for kernel
        density estimation. If a number is given, the number will be used as the bandwidth
        for kernel estimation.
    windows : int,
        The number of windows (segmentations) for KDE analysis. The default is 1, namely,
        the whole dataset is used for KDE analysis and gives the same result as using
        the function kdeScreen() directly.
    n_clusters : int
        The target number of cluster_center. The default is 20.
    x_test : Numpy array
        Test data to get the density distribution. The default is None.

    Returns
    -------
    kdeOutput : Numpy array
        A N x 3 dimension numpy array. The first column is the label of the window under analysis,
        the second is normlized distance, the third is the probability density.
    cluster_center : Numpy array
        A M x N dimension numpy array. M is the number of windows and N-1 is the number of cluster centers.
        The first column is the maximum index for each window, the following columns are the cluster centers.
    """
    import pandas as pd

    kdeOutput = np.zeros([dataset.shape[0], 3])
    cluster_center = np.zeros([windows - 1, n_clusters + 1])
    anchor = 0

    nslices = np.unique(dataset[:, -1]).size
    winLen = int((nslices / windows + 1))

    for win in range(0, windows - 1):
        # Point cloud in a window
        mask = (dataset[:, -1] < (win + 1) * winLen) & \
               (dataset[:, -1] >= win * winLen)
        variable = dataset[:, 0][mask].reshape(-1, 1)

        if x_test is None:
            # Generate test data to get the density distribution
            x_test = np.linspace(0, 1, variable.size)[:, np.newaxis]

        if type(bw).__module__ == "numpy":
            opt_bw = opt_bandwidth(variable, x_test, bw)
        elif str(int(bw * 1e20)).isdigit():
            opt_bw = bw
        else:
            print("Please check if bandwidth is given correctly!!!")

        # Call function kdeScreen() to get the variable-density curve
        # the index for cluster_center.
        variable = variable.flatten()
        clusters = kdeScreen(variable, x_test, opt_bw)

        # Get the index for cluster_center
        cluster_center_idx = clusters["cluster centers"]

        ykde = clusters["pdf"]

        print("The variable size is: ", x_test.size)
        upperLimit = anchor + x_test.size

        if len(cluster_center_idx) >= n_clusters:
            print(bcolors.ok("Window: {}:".format(win)))
            print("√ The required number of points {} was reached at h = {}. \
                  \nThe number of actual cluster_center is [{}]".format(
                n_clusters, round(opt_bw, 4), len(cluster_center_idx)))

            print("start index {}; end index {}; number of variables {}".format(
                anchor, upperLimit, upperLimit - anchor))

            print("The cluster centers are: ", cluster_center_idx)

            # sort the data from minimum to maximun and return the index
            maskSort = np.argsort(ykde[cluster_center_idx])
            extrDisordered = cluster_center_idx[maskSort][len(cluster_center_idx) - n_clusters:]

            # kdeOutput
            print("test", anchor, upperLimit)
            print(x_test.shape, ykde.shape)
            kdeOutput[anchor:upperLimit, 0] = win
            kdeOutput[anchor:upperLimit, 1] = x_test.flatten()
            kdeOutput[anchor:upperLimit, 2] = ykde.flatten()

            # cluster_center
            cluster_center[win, 0] = upperLimit
            cluster_center[win, 1:] = extrDisordered[np.argsort(extrDisordered)]
            anchor = upperLimit
        else:
            print("Window: {}:".format(win))
            print(bcolors.warning("--> Cannot reach the targeted {} points. \
                  There are [{}] points for h = {}. Please reduce bandwidth.\n"
                                  "--------------------".format(
                n_clusters, len(cluster_center_idx), round(opt_bw, 4))))

            kdeOutput[anchor:upperLimit, 0] = win
            cluster_center[win, 0] = upperLimit

            anchor = upperLimit
            continue

    kdeOutput_col = ["window", "normalized distance", "probability density"]
    kdeOutput = pd.DataFrame(kdeOutput, columns=kdeOutput_col)

    # Returns a column mask to show if any zero values are present in the row of kdeOutput
    # If any zero values are present, the row is masked (False), otherwise the row is not
    # masked (True). The mask is used to remove the rows with zero values from the kdeOutput.
    # the first row is always as True since it is the starting point (0) of the contour.
    mask = np.all(kdeOutput.iloc[:, [1, 2]] != 0, axis=1)
    mask[0] = True

    # remove the rows with zero values from the kdeOutput
    kdeOutput = kdeOutput[mask]

    return kdeOutput, cluster_center


def kdePlot(xkde, ykde, cluster_center_idx):
    """
    Parameters
    ----------
    xkde : Numpy array
        The normalized distance.
    ykde : Numpy array
        The probability density distribution corresponding to the normalized distance.
    cluster_center_idx : Numpy array
        The index of the cluster centers.

    Returns
    -------
        None.
    """
    fig = plt.figure(1, figsize=(12, 9))

    plt.close("all")
    plt.clf()
    plt.rcParams.update({'font.size': 16})

    plt.scatter(xkde[cluster_center_idx], ykde[cluster_center_idx])
    plt.plot(xkde, ykde)

    # Median
    cdf = scipy.integrate.cumtrapz(ykde, xkde, initial=0)
    nearest_05 = np.abs(cdf - cdf / 2).argmin()
    x_median, y_median = xkde[nearest_05], ykde[nearest_05]
    # Plot the median value as vertical line
    plt.vlines(x_median, 0, y_median, 'r')

    plt.legend()
    plt.xlabel('Normalized distance')
    plt.ylabel('Distribution density')

    # plt.savefig(str(windowIndex)+'.tiff')
    plt.show()
