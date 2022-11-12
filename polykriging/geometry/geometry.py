# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Polygon, Point


def angularSort(localCo, centroid, sort=True):
    """
    Sort the vertices of a 2D polygon in angular order. It can be a convex or concave polygon.

    Parameters
    ----------
    localCo : Numpy array
        with 2 colums. The x, y coordinate components of the vertices of the polygon (For the
        cross-section of fiber tows, it is the coordinate in the local coordinate system with
         its center at the centroid of the polygon).
    centroid : Numpy array
        with 2 colums. The x, y coordinate components of the centroid of the polygon.
    sort : Boolean
        If True, the vertices are sorted in angular order. If False, the vertices are
        not sorted and returned following the original order with angular position
        for each input vertices.

    Returns
    -------
    coorSort : Numpy array
        with 3 colums. The x, y coordinate components of the vertices of the polygon sorted in
        angular order. The third column is the z coordinate in 3d case.
    angle : Numpy array
        with 1 colums. The angular position of the vertices of the polygon in degree. The two
        returns are sorted in the same order if sort is True. Otherwise, the two returns are
        not sorted, and is following the original order of the input vertices.
    """
    # Angular positions of vertices in local coordinate. The origin is the centroid
    # of the cross-section
    xloc = localCo[:, 0]
    yloc = localCo[:, 1]
    angle = np.mod(np.arctan2(yloc, xloc), 2 * np.pi) * 180 / np.pi  # mapping to 0-360 degree

    minIndex = np.where(angle == np.min(angle))[0][0]
    maxIndex = np.where(angle == np.max(angle))[0][0]

    if abs(minIndex - maxIndex) == 1:
        pass
    elif maxIndex > minIndex:
        maxIndex = minIndex + 1
    else:
        maxIndex = maxIndex - 1

    coordinate = localCo[:,:2] + centroid[:2]

    if sort:
        coorSort = np.zeros([localCo.shape[0], 3])
        coorSort[:, :2] = np.append(coordinate[maxIndex:],
                                    coordinate[:maxIndex], axis=0)
        coorSort[:, 2] = centroid[2]
        angle = np.append(angle[maxIndex:], angle[:maxIndex], axis=0)

        if np.max(angle) in angle[:6]:
            coorSort = np.flip(coorSort, axis=0)
            angle = np.flip(angle, axis=0)

        # print("angle", angle, "max", np.max(angle), "min", np.min(angle))

        # origin
        xp = np.squeeze(localCo[[minIndex, maxIndex], 0])
        yp = np.squeeze(localCo[[minIndex, maxIndex], 1])
        # y is known.
        origin = [np.interp(0, yp, xp), 0, 0] + centroid
        coorSort = np.vstack((origin, coorSort))

        angle = np.hstack((0, angle))
    else:
        coorSort = np.zeros([coordinate.shape[0], 3])
        coorSort[:, :2] = coordinate
        coorSort[:, -1] = centroid[2]

    return coorSort, angle


# used in geom() function.
def edgeLen(localCo, boundType="rotated"):
    '''
    the width and height of rotated_rectangle
    boundType: rotated or parallel
    '''
    polygonLocal = Polygon(localCo)
    # Returns a (minx, miny, maxx, maxy) tuple that bounds the object.
    if boundType == "parallel":
        bounds = polygonLocal.bounds  # in parallel to axis
    elif boundType == "rotated":
        # Returns the general minimum bounding rectangle that contains the object.
        bounds = polygonLocal.minimum_rotated_rectangle
    else:
        print("Eror bound type. It can be parallel or rotated. The defalt is rotated. \
              \n The operation is killed.")
        import sys
        sys.exit()

    xb, yb = bounds.exterior.xy
    # print(np.array(xb)>0, np.array(yb)>0)

    if (str(np.array(xb) > 0) == '[ True  True False False  True]') or (
            str(np.array(xb) > 0) == '[ True False False  True  True]'):
        try:
            angleRotated = np.arctan((yb[0] - yb[1]) / (xb[0] - xb[1])) / np.pi * 180
        except:
            angleRotated = 0
    elif str(np.array(xb) > 0) == '[False  True  True False False]':
        try:
            angleRotated = np.arctan((yb[1] - yb[0]) / (xb[1] - xb[0])) / np.pi * 180
        except:
            angleRotated = 0
    else:
        angleRotated = 0

    # get length of bounding box edges (a rectanglar)
    edge_length = (Point(xb[0], yb[0]).distance(Point(xb[1], yb[1])),
                   Point(xb[1], yb[1]).distance(Point(xb[2], yb[2])))
    # get length of polygon as the longest edge of the bounding box
    width = max(edge_length)
    # get height of polygon as the shortest edge of the bounding box
    height = min(edge_length)

    return width, height, angleRotated


def normDist(localCo):
    '''
    The normalized distance of the vertices of a polygon
    '''
    distance = np.zeros([len(localCo)])
    for i in np.arange(localCo.shape[0] - 1):
        distance[i + 1] = np.linalg.norm(localCo[i + 1] - localCo[i]) + distance[i]

    # normalization
    normDistance = distance / np.max(distance)
    return distance, normDistance


def geom_cs(coordinate, message="OFF", sort=True):
    """
    Geometry analysis and points sorting for a cross-section of a fiber tow.

    Parameters
    ----------
    coordinate : Numpy array
        with 3 colums. The x, y, z coordinate components of the vertices of the polygon.
        Note: 只使用了前两个

    Returns
    -------
    geometry file : x,y,z of points, and x,y,z of centerline
    properties: area... ...

    """

    polygon = Polygon(coordinate[:, [0, 1]])

    # Area, Perimeter and Circularity.
    area = polygon.area  # Area
    perimeter = polygon.length  # Perimeter
    Circularity = 4 * np.pi * area / np.square(perimeter)  # Circularity

    # Centroid 2D in string format
    centroid_wkt = polygon.centroid.wkt
    centroid_2d = np.fromstring(centroid_wkt[7:len(centroid_wkt) - 1], dtype=float, sep=" ")
    centroid = np.hstack((centroid_2d, coordinate[0, -1]))

    # local coordinates of the cross-section
    localCo = coordinate[:, [0, 1]] - centroid.take([0, 1])

    # width, height, angleRotated
    width, height, angleRotated = edgeLen(localCo, boundType="rotated")

    coordinateSorted, anglePosition = angularSort(localCo, centroid, sort)

    # To close the polygon:
    coordinateSorted = np.vstack((coordinateSorted, coordinateSorted[0, :]))
    anglePosition = np.append(anglePosition, 360)

    localCo = coordinateSorted[:, [0, 1]] - centroid.take([0, 1])
    distance, normDistance = normDist(localCo)

    # [distance, normalized distance, angular position (degree), coordinateSorted(X, Y, Z)]
    coordinateSorted = np.hstack((distance.reshape([-1, 1]),
                                  normDistance.reshape([-1, 1]),
                                  anglePosition.reshape([-1, 1]),
                                  coordinateSorted))

    # [Area, Perimeter, Width, Height, AngleRotated, Circularity,
    #   centroidX, centroidY, centroidZ]
    geomFeature = np.hstack((area, perimeter, width, height, angleRotated, Circularity, centroid))

    if message == "ON":
        print(" The geometrical features of this cross-section: \
    \n [Area, Perimeter, Width, Height, AngleRotated, Circularity, centroidX, centroidY, centroidZ]")

        print(geomFeature)

    return geomFeature, coordinateSorted


def geom_tow(surf_points, sort=True):
    """
    The surface points for each cross-sections. the last column (z-axis) should be along the extension
    direction of the cross-sections. It also serves as the label of each cross-section.

    Parameters
    ----------
    surf_points : array_like
        The surface points for each cross-sections. the last column (z-axis) should be along the extension
        direction of the cross-sections. It also serves as the label of each cross-section.

    Returns
    -------
    df_geom : DataFrame
        The geometrical features of each cross-sections. The columns are:
        [Area, Perimeter, Width, Height, AngleRotated, Circularity, centroidX, centroidY, centroidZ]
    df_coo : DataFrame
        The coordinates of each cross-sections. The columns are:
        [distance, normalized distance, angular position (degree), X, Y, Z)]
    """
    slices = np.unique(surf_points[:, -1])
    centerline = np.zeros([slices.size, 3])
    for iSlice in range(slices.size):
        # for iSlice in range(2):
        mask_slice = surf_points[:, -1] == slices[iSlice]
        coordinate = surf_points[mask_slice, -3:]
        surf_points = surf_points[~mask_slice, :]

        geomFeature, coordinateSorted = geom_cs(coordinate, sort=sort)

        centerline[iSlice - 1, :] = geomFeature[-3:]

        try:
            geomFeatures = np.vstack((geomFeatures, geomFeature))
            coordinatesSorted = np.vstack((coordinatesSorted, coordinateSorted))
        except NameError:
            geomFeatures = geomFeature
            coordinatesSorted = coordinateSorted

    column_geom = ["Area", "Perimeter", "Width", "Height", "AngleRotated", "Circularity",
                   "centroidX", "centroidY", "centroidZ"]
    column_coord = ["distance", "normalized distance", "angular position (degree)",
                    "X", "Y", "Z"]

    import pandas as pd

    # recover the original order
    df_geom = pd.DataFrame(geomFeatures, columns=column_geom)
    df_coord = pd.DataFrame(coordinatesSorted, columns=column_coord)

    return df_geom, df_coord


if __name__ == "__main__":
    resolution = 0.022
    coordinate = np.load("arr_1.npy")[:, 1:] * resolution
    geomFeature, coordinateSorted = geom(coordinate)
    # coorSort, angle = angularSort(localCo, centroid)
