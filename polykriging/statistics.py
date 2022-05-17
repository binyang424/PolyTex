import numpy as np
import scipy
from scipy.signal import argrelextrema
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def optBandwidth(variable, x_test, bw):
    '''
    Find the optimal bandwidth by tuning of the `bandwidth` parameter via cross-validation and returns
    the parameter value that maximizes the log-likelihood of data.
    '''
    
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, {'bandwidth': bw})
    grid.fit(variable)
    
    kde = grid.best_estimator_
    log_dens = kde.score_samples(x_test)
    print("optimal bandwidth: " + "{:.4f}".format(kde.bandwidth))

    return kde.bandwidth


def kdeScreen(variable, x_test, bw, kernels = 'gaussian', plot = "False"):

    model = KernelDensity(kernel = kernels, bandwidth=bw)
    model.fit(variable)
    log_dens = model.score_samples(x_test)       
    kde = plt.plot(x_test, np.exp(log_dens), c='cyan')
    # xkde: normalized distance; ykde: density
    xkde, ykde = kde[0].get_data()
    if plot != "False":
        plt.show()
        
    # mask for the local maxima of density
    extremaIndex = argrelextrema(ykde, np.greater)[0]
    # print(extremaIndex.size, "\n", extremaIndex)
    
    return xkde, ykde, extremaIndex


def movingKDE(variable, bandwidth, windowIndex = 1, extremaNum=20):
    '''
    Plot univariate or bivariate distributions using kernel density estimation.
    :param

    '''
    for win in np.arange(1, windows):
        # Point cloud in a window
        maskMax = pcd[:,-1] < (win + 1) * winLen * 0.022
        variable = pcd[:,1][maskMax]
        maskMin = pcd[:,-1][maskMax] > win * winLen * 0.022
        variable = variable[maskMin].reshape(-1, 1)

        # Generate test data to get the density distribution
        x_test = np.linspace(0, 1, variable.size)[:, np.newaxis]

        print("Window: ", win)
        #optBw = optBandwidth(variable, x_test, bw)

        optBw = 0.002
        xkde, ykde, extremaIndex = kdeScreen(variable, x_test, optBw)

    return xkde, ykde, extremaIndex


def kdePlot():


    fig = plt.figure(1, figsize=(12, 9))

    plt.close("all")
    plt.clf()
    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 16})


    plt.scatter(xkde[extremaIndex], ykde[extremaIndex])

    # Median
    cdf = scipy.integrate.cumtrapz(ykde, xkde, initial=0)
    nearest_05 = np.abs(cdf - cdf/2).argmin()
    x_median, y_median = xkde[nearest_05], ykde[nearest_05]
    # Plot the median value as vertical line
    plt.vlines(x_median, 0, y_median, 'r')

    plt.legend()
    plt.xlabel('Normalized distance')
    plt.ylabel('Distribution density')

    #plt.savefig(str(windowIndex)+'.tiff')
    plt.show()



if __name__ == "__main__":
    path = r"C:\BinY\DMT\Code\processedData\Vf60\warp\yarn_1.npz"
    path = r"C:\Users\palme\Desktop\Code\processedData\Vf60\warp\yarn_1.npz"
    pcdRaw = np.load(path)
    fileList = list(pcdRaw.keys())
    yarn = 1
    
##    for i in range(2):
    for i in range(len(fileList) - 1):
        slicedata = "yarn_" + str(yarn) + "_coordinateSorted_" + str(i) + ".npy"
        try:
            pcd = np.vstack((pcd,pcdRaw[slicedata]))
        except NameError:
            pcd = pcdRaw[slicedata]
##        if i == 0:
##            fig = plt.figure()
##            ax = fig.add_subplot(projection='polar')
##            # ax.set_ylabel('Normalized distance')
##        # The following angle positions should be in radians.
##        ax.scatter(pcdRaw[slicedata][:, 2]/360*2*np.pi, pcdRaw[slicedata][:,1],
##                   alpha = 0.7, s = 1.5 )
##    # reference line for a circle:
##    ax.plot(np.arange(0, 2*np.pi, 2*np.pi/360), np.arange(0,1,1/360), linestyle='--', color = 'red' )      
##    # reference line for a square:
##    
##    #plt.show()

    # geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
    #       centroidX, centroidY, centroidZ]
    pcd = pcd    
    # [distance, normalized distance, angular position (degree),
    #       coordinateSorted(X, Y, Z)]
    geomFeatures = pcdRaw["yarn_1_geomFeatures.npy"]


    #############################################################
    #                 KDE analysis
    #############################################################   

    ### 分段：
    windows = 18
    winLen = int( ( len(fileList) - 1 )/ windows + 1 )
    bw = np.arange(0.005, 0.009, 0.0003)
    extremaNum = 15


    movingKDE
##    for win in np.arange(1, windows):
##        # Point cloud in a window
##        maskMax = pcd[:,-1] < (win + 1) * winLen * 0.022
##        variable = pcd[:,1][maskMax]
##        maskMin = pcd[:,-1][maskMax] > win * winLen * 0.022
##        variable = variable[maskMin].reshape(-1, 1)
##
##        # Generate test data to get the density distribution
##        x_test = np.linspace(0, 1, variable.size)[:, np.newaxis]
##
##        print("Window: ", win)
##        #optBw = optBandwidth(variable, x_test, bw)
##
##        optBw = 0.002
##        xkde, ykde, extremaIndex = kdeScreen(variable, x_test, optBw)

        if len(extremaIndex) > extremaNum:
            print("The required number of points ({}) was reached at h = {}. \
                  \nThe number of actual extrema is {}. \
                  --------------------".format(
                extremaNum, round(optBw, 4), len(extremaIndex)) )

            maskSort = np.argsort(ykde[extremaIndex]) < extremaNum
            xextrema = xkde[extremaIndex][maskSort]
            yextrema = ykde[extremaIndex][maskSort]
        else:
            print("Cannot reach the targeted {} points.There are {} points for h = {}. \
                  --------------------".format(
                extremaNum, len(extremaIndex), round(optBw, 4)))

    
##    plt.close("all")
##    plt.clf()
##    plt.plot(xkde, ykde)
##    plt.show()
