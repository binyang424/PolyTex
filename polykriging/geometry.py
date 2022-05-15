# -*- coding: utf-8 -*-

import numpy as np
from shapely.geometry import Polygon, Point

# used in geom() function.
def angularSort(localCo, centroid):
    '''
    input: local coordinate of polygon vertices for angular sorting
    output: angle position and sorted coordinate according angle position
    '''  
    # Angular positions of vertices in local coordinate. The origin is the centroid
    # of the cross-section
    xloc = localCo[:, 0]
    yloc = localCo[:, 1]
    angle=np.mod(np.arctan2(yloc, xloc), 2*np.pi)*180/np.pi   # mapping to 0-360 degree

    minIndex = np.where(angle == np.min(angle))[0][0]
    maxIndex = np.where(angle == np.max(angle))[0][0]
    # print(localCo[minIndex], localCo[maxIndex])

    xp = np.squeeze(localCo[[minIndex,maxIndex], 0])
    yp = np.squeeze(localCo[[minIndex,maxIndex], 1])

    # y is known.
    origin = [np.interp(0, yp, xp), 0, 0] + centroid
    
    coordinate = localCo + centroid[:2]
    coorSort = np.zeros([localCo.shape[0], 3])
    coorSort[:, :2] = np.append(coordinate[maxIndex:], coordinate[:maxIndex], axis=0)
    coorSort[:, 2] = centroid[2]
    angle = np.append(angle[maxIndex:], angle[:maxIndex], axis=0)
    if np.min(angle) == angle[-1]:
        coorSort = np.flip(coorSort,axis=0)
        angle = np.flip(angle,axis=0)

    angle = np.hstack((0, angle))
    coorSort = np.vstack((origin, coorSort))
    
    return coorSort, angle


# used in geom() function.
def edgeLen(localCo, boundType = "rotated"):
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
    edge_length = ( Point(xb[0], yb[0]).distance(Point(xb[1], yb[1]) ),
                    Point( xb[1], yb[1] ).distance( Point( xb[2], yb[2]) ) )
    # get length of polygon as the longest edge of the bounding box
    width = max(edge_length)
    # get height of polygon as the shortest edge of the bounding box
    height = min(edge_length)
    
    return width, height, angleRotated


# 分析2D的截面信息
def geom(coordinate, message = "OFF"):
    '''
    Parameters
    ----------
    coordinate : Numpy array with 3 colums. The x, y, z coordinate components of
    polygon vertices in 3D Cartesian system are stored.

    只使用了前两个

    Returns
    -------
    geometry file: x,y,z of points, and x,y,z of centerline
    properties: area... ... 
    
    '''
    #global localCo, centroid, coordinateSorted, anglePosition, width, height, angleRotated
    
    polygon = Polygon(coordinate[:, [0, 1]])
    
    area = polygon.area    # Area 
    perimeter = polygon.length   # Perimeter

    # Centroid 2D in string format
    centroid_wkt = polygon.centroid.wkt
    centroid_2d = np.fromstring(centroid_wkt[7:len(centroid_wkt) - 1], dtype=float, sep=" ")
    centroid = np.hstack((centroid_2d, coordinate[0, -1]))

    # local coordinates of the cross-section
    localCo = coordinate[:, [0, 1]] - centroid.take([0, 1])

    # width, height, angleRotated
    width, height, angleRotated = edgeLen(localCo, boundType = "rotated")
    coordinateSorted, anglePosition = angularSort(localCo, centroid)

    Circularity = 4 * np.pi * area / np.square(perimeter)
    # [Area, Perimeter, Width, Height, AngleRotated, Circularity,
    #   centroidX, centroidY, centroidZ]
    geomFeature = np.hstack((area, perimeter, width, height, angleRotated, Circularity, centroid))

    if message == "ON":
        print(" The geometrical features of this cross-section: \
    \n [Area, Perimeter, Width, Height, AngleRotated, Circularity, centroidX, centroidY, centroidZ]")

        print(geomFeature)
    
    return  geomFeature, coordinateSorted, anglePosition


if __name__ == "__main__":
    resolution = 0.022
    coordinate = np.load("coordinate.npy")[:,1:] * resolution
    geomFeature, coordinateSorted, anglePosition = geom(coordinate)
