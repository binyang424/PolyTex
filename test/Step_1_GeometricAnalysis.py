# -*- coding: utf-8 -*-
"""
Step 1: GeometricAnalysis
=========================

Input: pcd
Output: .geo and .coo files
"""

import numpy as np
import polytex as pk
from polytex.geometry import geom
import matplotlib.pyplot as plt
import pandas as pd

'''
Variables:
resolution - the resolution of micro CT scan (size of voxels)
yarn - the number of the yarn
surfPoints - df of raw point cloud dataset: [original point order, X, Y, Z (SLICE NUMBER)]
'''

# the path where the npz file is stored.
# ./05_polytex\Data\extractedData\
path = pk.io.choose_directory(
    titl="Choose the directory that contains raw point cloud data (pcd) ...")

pk.io.cwd_chdir(path)  # set the path as current work directory
filelist = pk.io.filenames(path, "pcd")

pathsave = pk.io.choose_directory(
    titl="Choose the directory to save Geometry features")

resolution = 0.022  # mm/pixel

fig = plt.figure()
for yarn in np.arange(0, 7):
    try:
        # load contour described by point cloud
        pcd = pk.pk_load("weft_{}.pcd".format(yarn))
    except FileNotFoundError:
        print("weft_{}.pcd not found!".format(yarn))
        continue

    surfPoints = pcd.to_numpy()[:, 1:] * resolution

    slices = np.unique(surfPoints[:, -1]) /resolution
    nslice = slices.size
    centerline = np.zeros([nslice, 3])

    for iSlice in range(slices.size):

        coordinate = surfPoints[surfPoints[:, -1] == slices[iSlice]*resolution, -3:]

        # geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
        #       centroidX, centroidY, centroidZ]
        # coordinateSorted = [distance, normalized distance, angular position (degree),
        #       coordinateSorted(X, Y, Z)]
        geomFeature, coordinateSorted = geom(coordinate, sort=True)
        centerline[iSlice - 1, :] = geomFeature[-3:]

        try:
            geomFeatures = np.vstack((geomFeatures, geomFeature))
            coordinatesSorted = np.vstack((coordinatesSorted, coordinateSorted))
        except NameError:
            geomFeatures = geomFeature
            coordinatesSorted = coordinateSorted

        # plot the contours and rotated boxes
        # close all the figures

        if iSlice % 13 == 0:
            ax = fig.add_subplot(13, 8, int(iSlice / 15 + 1))
            ax.set_axis_off()
            plt.fill(coordinateSorted[:, 3], coordinateSorted[:, 4], facecolor='pink', alpha=0.5)
            # plt.plot(xb,yb)   # plt.plot(*polygon.exterior.xy)  # Error on the last iSlice
            plt.scatter(geomFeature[-3], geomFeature[-2], marker='.', c='r')
            ax = plt.gca()
            ax.set_aspect(1)
            # plt.axis('off')
    plt.show()

    columns_geom = ["Area", "Perimeter", "Width", "Height", "AngleRotated", "Circularity",
          "centroidX", "centroidY", "centroidZ"]
    columns_coor = ["distance", "normalized distance", "angular position (degree)",
          "X", "Y", "Z"]

    df_geom = pd.DataFrame(geomFeatures, columns=columns_geom)
    df_coor = pd.DataFrame(coordinatesSorted, columns=columns_coor)

    # save the geomFeature properties

    pk.pk_save(pathsave + "\\weft_" + str(yarn) + ".geo", df_geom)
    pk.pk_save(pathsave + "\\weft_" + str(yarn) + ".coo", df_coor)

    del surfPoints, coordinate, geomFeature, coordinateSorted, geomFeatures, coordinatesSorted
    plt.close('all')