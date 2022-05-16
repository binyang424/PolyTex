import numpy as np
import scipy
from scipy.signal import argrelextrema
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
import matplotlib.pyplot as plt



def movingKDE(variable, bandwidth, windowIndex, extremaNum=50):
    """
    # Plot univariate or bivariate distributions using kernel density estimation.
    :param variable:
    :param bandwidth:
    :param extremaNum:
    :return:
    """
    import seaborn as sns

    fig = plt.figure(1, figsize=(12, 9))

    plt.close("all")
    plt.clf()
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 16})

    Count = 0  # get the curve data (index of kde curves)
    for h in bandwidth:
        kde = sns.kdeplot(variable, cut=3, clip=None, bw_adjust=h, label='h = ' + str(h))  # Gaussian kernel
        xkde, ykde = kde.get_lines()[Count].get_data()

        # local maxima
        lstmax = argrelextrema(ykde, np.greater)  # import from scipy.signal

        if len(lstmax[0]) > extremaNum:
            print("The required number of points ({}) was reached at h = {}.".format(
                extremaNum, round(h, 4)))

            maskSort = np.argsort(ykde[lstmax]) < extremaNum
            xextrema = xkde[lstmax][maskSort]
            yextrema = ykde[lstmax][maskSort]

            break
        elif h == bandwidth[-1]:
            print("Cannot reach the targeted {} points.There are {} points for h = {}.".format(
                extremaNum, len(lstmax[0]), round(h, 4)))
        Count += 1

    plt.scatter(xkde[lstmax], ykde[lstmax])

    # Median
    cdf = scipy.integrate.cumtrapz(ykde, xkde, initial=0)
    nearest_05 = np.abs(cdf - 0.5).argmin()
    x_median, y_median = xkde[nearest_05], ykde[nearest_05]
    # plt.vlines(x_median, 0, y_median, 'r')

    plt.legend()
    plt.xlabel('Normalized distance')
    plt.ylabel('Distribution density')

    plt.savefig(str(windowIndex)+'.tiff')
    # plt.show()

    return xextrema, yextrema


if __name__ == "__main__":
    
    pcdRaw = np.load(
        r"C:\BinY\DMT\Code\processedData\Vf60\warp\yarn_1.npz")
    fileList = list(pcdRaw.keys())
    yarn = 1
    for i in range(2):
##    for i in range(len(fileList) - 1):
        slicedata = "yarn_" + str(yarn) + "_coordinateSorted_" + str(i) + ".npy"
        try:
            pcd = np.vstack((pcd,pcdRaw[slicedata]))
        except NameError:
            pcd = pcdRaw[slicedata]
        
        plt.scatter(pcdRaw[slicedata][:,1], pcdRaw[slicedata][:, 2], alpha = 0.8, s = 1 )
        
    # geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
    #       centroidX, centroidY, centroidZ]
    # coordinateSorted = [distance, normalized distance, angular position (degree),
    #       coordinateSorted(X, Y, Z)]
    geomFeatures = pcdRaw["yarn_1_geomFeatures.npy"]
        
    plt.show()
