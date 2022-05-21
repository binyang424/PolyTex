# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Polygon, Point
from polykriging.utility import cwd_chdir, filenames
from polykriging.geometry import geom
import matplotlib.pyplot as plt

'''
Variables:
    resolution - the resolution of micro CT scan (size of voxels)
    yarn - the number of the yarn
    yarnDict - the dictionary file (.npz) storing the point of fiber yarn
'''

# the path where the npz file is stored.
##path = r"C:\Users\palme\Desktop\Code\rawPointClouds\Vf60\warp"
path = r"C:\BinY\DMT\Code\rawPointClouds\Vf60\warp"

cwd_chdir(path)  #set the path as current work directory
filelist = filenames(path, "npz")

resolution = 0.022  # mm/pixel

#fig = plt.figure()
for yarn in np.arange(1,2):
    yarnDict = dict(np.load("warp_" + str(yarn) + ".npz"))
    # load contour described by point cloud
   
    nslice = len(yarnDict)
    centerline = np.zeros([nslice, 3])
    for iSlice in np.arange(nslice):
        coordinate = yarnDict["arr_" + str(iSlice)][:,1:] * resolution

        # geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
        #       centroidX, centroidY, centroidZ]
        # coordinateSorted = [distance, normalized distance, angular position (degree),
        #       coordinateSorted(X, Y, Z)]
        geomFeature, coordinateSorted = geom(coordinate)
        centerline[iSlice, :] = geomFeature[-3:]

        # save the sorted cross-sectional points of each slice
        np.save(r"C:\BinY\DMT\Code\processedData\Vf60\warp\yarn_" + str(yarn)
                + "_coordinateSorted_" + str(iSlice), coordinateSorted)

        try:
            geomFeatures = np.vstack((geomFeatures, geomFeature))
        except NameError:
            geomFeatures = geomFeature
    
##        # plot the contours and rotated boxes
##        if iSlice % 10 == 0:
##            ax = fig.add_subplot(10, 6, int(iSlice / 10+ 1))
##            ax.set_axis_off()
##            plt.fill(coordinateSorted[:, 3], coordinateSorted[:,4], facecolor='pink', alpha=0.5)
##            #plt.plot(xb,yb)   # plt.plot(*polygon.exterior.xy)  # Error on the last iSlice
##            plt.scatter(geomFeature[-3], geomFeature[-2], marker='.', c='r')
##            ax = plt.gca()
##            ax.set_aspect(1)
##            #plt.axis('off')
##    plt.show()

    # save the geomFeature properties
    np.save(r"C:\BinY\DMT\Code\processedData\Vf60\warp\yarn_" + str(yarn) + "_geomFeatures", geomFeatures)

    ## Note: The .npy files will be compressed and stored in a .npz file
    ## in the name of yarn_[yarn number].npz.
