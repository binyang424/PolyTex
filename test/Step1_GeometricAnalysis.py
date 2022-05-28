# -*- coding: utf-8 -*-

import numpy as np
from polykriging.utility import cwd_chdir, filenames, choose_directory
from polykriging.geometry import geom
import matplotlib.pyplot as plt

'''
Variables:
    resolution - the resolution of micro CT scan (size of voxels)
    yarn - the number of the yarn
    surfPoints - the numpy array file (.npy) storing the point of fiber yarn in the format of
                        [original point order, X, Y, Z (SLICE NUMBER)]
'''

# the path where the npz file is stored.
path = choose_directory(titl =
                                "Choose the directory that contains surfPoints")

cwd_chdir(path)  #set the path as current work directory
filelist = filenames(path, "npy")

resolution = 1  # mm/pixel

fig = plt.figure()
for yarn in np.arange(1,2):
    # surfPoints = np.load("weft_" + str(yarn) + ".npy")
    surfPoints = np.load("PointsWithError3.0Percent.npy")
    # load contour described by point cloud

    slices = np.unique(surfPoints[:, -1])
    nslice = slices.size
    centerline = np.zeros([nslice, 3])
    
    for iSlice in range(slices.size):

        coordinate = surfPoints[surfPoints[:, -1] == slices[iSlice], -3:] * resolution

        # geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
        #       centroidX, centroidY, centroidZ]
        # coordinateSorted = [distance, normalized distance, angular position (degree),
        #       coordinateSorted(X, Y, Z)]
        geomFeature, coordinateSorted = geom(coordinate, sort = False)
        centerline[iSlice-1, :] = geomFeature[-3:]

        try:
            geomFeatures = np.vstack((geomFeatures, geomFeature))
            coordinatesSorted = np.vstack((coordinatesSorted, coordinateSorted))
        except NameError:
            geomFeatures = geomFeature
            coordinatesSorted = coordinateSorted
    
        # plot the contours and rotated boxes
        if iSlice % 10 == 0:
            ax = fig.add_subplot(10, 6, int(iSlice / 1 + 1))
            ax.set_axis_off()
            plt.fill(coordinateSorted[:, 3], coordinateSorted[:,4], facecolor='pink', alpha=0.5)
            #plt.plot(xb,yb)   # plt.plot(*polygon.exterior.xy)  # Error on the last iSlice
            plt.scatter(geomFeature[-3], geomFeature[-2], marker='.', c='r')
            ax = plt.gca()
            ax.set_aspect(1)
            #plt.axis('off')
    plt.show()

    # save the geomFeature properties
    pathsave = choose_directory(titl =
                                "Choose the directory to save Geometry features")
    np.savez(pathsave + "\\yarn_" + str(yarn),
             geomFeatures = geomFeatures, coordinatesSorted = coordinatesSorted)

