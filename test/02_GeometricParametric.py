# -*- coding: utf-8 -*-

"""
Bin Yang
September, 2021
Polytechnique Montreal and WHUT

1. Sorting all the feature points of each contour from the minimum angular position
2. Calculate the normalised distance and save with angular position
3. Save parametric Kriging expression (x, y as a function of normalized distance)
"""

import csv, os
import numpy as np
from polykriging import curve2D, utility


##########################################
#####            Execution           #####
##########################################

direction = 'warp'
# linear density
rho_yarn = {"binder": 275, "warp": 2200, "weft": 1100}  # tex, g/km

contourKrigExpr = {}  # the parametric curve kriging function for each contour

for yarnIndex in np.arange(30):


    nslice = len(fcsv)
    contourKrigExprSlice = {}
    for iSlice in np.arange(nslice):  # nslice
        
        """ Local coordinate """
        # weft tows
        localCo = (coordinate - centroid)[:, [0, 1]]





        nugget = 1e-15

        xmat_krig, xmat_krig_inv, xvector_ba, xexpr = curve2D.curveKrig(geomParametricSlice[:, [5, 1]],
                                                                        'lin', 'lin', nuggetEffect=nugget)
        ymat_krig, ymat_krig_inv, yvector_ba, yexpr = curve2D.curveKrig(geomParametricSlice[:, [5, 3]],
                                                                        'lin', 'lin', nuggetEffect=nugget)

        sliceKey = str(iSlice)
        contourKrigExprSlice[sliceKey] = [str(xexpr), str(yexpr)]

        print(yarnIndex, '--', sliceKey)

        try:
            geomParametric = np.vstack((geomParametric, geomParametricSlice))
        except NameError:
            geomParametric = geomParametricSlice

    yarnKey = str(yarnIndex)
    contourKrigExpr[yarnKey] = contourKrigExprSlice
