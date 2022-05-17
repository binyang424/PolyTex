import numpy as np
import scipy
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
    '''
    

    Parameters
    ----------
    variable : Numpy array
        A N x 1 dimension numpy array.
    x_test : Numpy array 
        Test data to get the density distribution. 
        It has the same shape as the given variable.
    bw : float
        DESCRIPTION.
    kernels : TYPE, optional
        DESCRIPTION. The default is 'gaussian'.
    plot : TYPE, optional
        DESCRIPTION. The default is "False".

    Returns
    -------
    xkde : TYPE
        DESCRIPTION.
    ykde : TYPE
        DESCRIPTION.
    extremaIndex : TYPE
        DESCRIPTION.

    '''
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


def movingKDE(pcd, bw = 0.002, windows = 1, extremaNum=20):
    '''
    Parameters
    ----------
    pcd : Numpy array
        A N x 2 dimension numpy array for kernel density estimation.
        The first colum should be the variable under analysis, the second
        is the tag of cross-sections for windows separation.
    bw : Numpy array or a number
        - A range of bandwidth values for kde operation usually generated with np.arange().
        The optimal bandwidth will be identified within this range and be used for kernel
        density estimation.
        - If a number is given, the number will be used as the bandwidth for kernel estimation.
    windows : int, The number of windows (segmentations) for KDE analysis
        DESCRIPTION. The default is 1.
    extremaNum : int, The target number of extrema
        DESCRIPTION. The default is 20.

    Returns
    -------
    xkde : TYPE
        DESCRIPTION.
    ykde : TYPE
        DESCRIPTION.
    extremaIndex : TYPE
        DESCRIPTION.

    '''
    kdeOutput = np.zeros([pcd.shape[0],3])
    extrema = np.zeros([windows, extremaNum + 1])
    anchor = 0
    for win in np.arange(0, windows):
        # Point cloud in a window
        maskMax = pcd[:,-1] < (win + 1) * winLen * 0.022
        variable = pcd[:,1][maskMax]
        # Do not miss "=".
        maskMin = pcd[:,-1][maskMax] >= win * winLen * 0.022
        variable = variable[maskMin].reshape(-1, 1)

        # Generate test data to get the density distribution
        x_test = np.linspace(0, 1, variable.size)[:, np.newaxis]
      
        if type(bw).__module__ == "numpy":
            optBw = optBandwidth(variable, x_test, bw)
        elif str(int(bw*1e20)).isdigit():
            optBw = bw
        else:
            print("Please check if a bandwidth is given correctly!!!")
        
        
        # Call function kdeScreen() to get the variable-density curve
        # the index for extrema.
        xkde, ykde, extremaIndex = kdeScreen(variable, x_test, optBw)
        
        upperLimit = anchor + variable.size
        
        print(anchor, upperLimit, upperLimit - anchor)
        
        if len(extremaIndex) > extremaNum:
            print("Window: {}:".format(win) )
            print("√ The required number of points {} was reached at h = {}. \
                  \nThe number of actual extrema is [{}]. \
                  --------------------".format(
                extremaNum, round(optBw, 4), len(extremaIndex)) )
            
            # sort the data from minimum to maximun and return the index
            maskSort = np.argsort(ykde[extremaIndex])
            
            # kdeOutput
            kdeOutput[anchor:upperLimit, 0] =  win
            kdeOutput[anchor:upperLimit, 1] =  xkde
            kdeOutput[anchor:upperLimit, 2] =  ykde
            
            # extrema
            extrema[win, 0] = upperLimit
            extrema[win, 1:] = extremaIndex[maskSort][
                len(extremaIndex)-extremaNum:]
            anchor = upperLimit
        else:
            print(f"{bcolors.WARNING}Window: {bcolors.ENDC}", (win) )
            print(" --> Cannot reach the targeted {} points. \
                  There are [{}] points for h = {}. \
                  Please reduce bandwidth.\n--------------------".format(
                    extremaNum, len(extremaIndex), round(optBw, 4)))
    
            kdeOutput[anchor:upperLimit, 0] =  win
            extrema[win, 0] = upperLimit
            
            anchor = upperLimit
            continue
    return kdeOutput, extrema


def kdePlot(xkde, ykde, extremaIndex):


    fig = plt.figure(1, figsize=(12, 9))

    plt.close("all")
    plt.clf()
    plt.rcParams.update({'font.size': 16})


    plt.scatter(xkde[extremaIndex], ykde[extremaIndex])
    plt.plot(xkde, ykde)

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
    #path = r"C:\Users\palme\Desktop\Code\processedData\Vf60\warp\yarn_1.npz"
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

    # KDE analysis

    windows = 10
    winLen = int( ( len(fileList) - 1 )/ windows + 1 )
    bw = 0.002
    # bw = np.arange(0.002, 0.01, 0.001)
    extremaNum = 15

    kdeOutput, extrema = movingKDE(pcd, bw, windows, extremaNum=35)

    del i, winLen, yarn, slicedata, pcdRaw, path, bw
    