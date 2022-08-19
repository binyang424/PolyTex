import time
from polykriging import utility
import numpy as np
from shapely.geometry import Polygon, Point
import pyvista as pv


# def isInbBox(bbox, point):
#     """
#     Determine if point is within a bounding box (bbox)
#     that parallel to the principle axes of global Cartesian coordinate.
#
#     :param bbox: bounding box, list: [xmin, xmax, ymin, ymax, zmin, zmax]
#     :param point: point, list: [x, y, z]
#     :return: True or False
#     """
#     x, y, z = point[0], point[1], point[2]
#     xmin, ymin, zmin = bbox[0], bbox[1], bbox[2]
#     xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]
#     boolst = [True, True, True, False, False, False]
#     boo = [x > xmin, y > ymin, z > zmin, x > xmax, y > ymax, z > zmax]
#     return boo == boolst


def background_mesh_generator(bbox, voxel_size=None):
    """
    Generate a voxel background mesh.
    :param bbox: bounding box of the background mesh specified through a numpy array
        contains the minimum and maximum coordinates of the bounding box
        [xmin, xmax, ymin, ymax, zmin, zmax]
    :param voxel_size: voxel size of the background mesh, type: None, float, or numpy.ndarray
        if None, the voxel size is set to the 1/20 of the diagonal length of the bounding box;
        if float, the voxel size is set to the float value in x, y, z directions;
        if numpy.ndarray, the voxel size is set to the values in the numpy.ndarray for corresponding directions.
    :return: pyvista mesh object (UnstructuredGrid)
    """
    # get the size of the bounding box
    size = np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]])
    # get the diagonal length of the bounding box
    diagonal = np.linalg.norm(size)

    if voxel_size is None:
        # Return (1/20) length of the diagonal of the bounding box.
        voxel_size_x, voxel_size_y, voxel_size_z = diagonal / 20, diagonal / 20, diagonal / 20
    if isinstance(voxel_size, (int, float)):
        voxel_size_x, voxel_size_y, voxel_size_z = voxel_size, voxel_size, voxel_size
    if isinstance(voxel_size, (list, set, tuple)):
        voxel_size_x, voxel_size_y, voxel_size_z = voxel_size

    xrng = np.linspace(bbox[0], bbox[1], int((bbox[1] - bbox[0]) / voxel_size_x) + 1)
    yrng = np.linspace(bbox[2], bbox[3], int((bbox[3] - bbox[2]) / voxel_size_y) + 1)
    zrng = np.linspace(bbox[4], bbox[5], int((bbox[5] - bbox[4]) / voxel_size_z) + 1)

    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pv.StructuredGrid(x, y, z)

    return grid.cast_to_unstructured_grid()


def find_cells_within_bounds(mesh, bounds):
    from pyvista import _vtk
    from pyvista import vtk_id_list_to_array
    """Find the index of cells in this mesh within bounds.
    :param mesh: pyvista mesh
    :param bounds: type:　iterable(float)
        list of 6 values, [xmin, xmax, ymin, ymax, zmin, zmax]
    :return: type: numpy.ndarray
        array of cell indices
    """
    if np.array(bounds).size != 6:
        raise TypeError("Bounds must be a length three tuple of floats.")
    locator = _vtk.vtkCellTreeLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    id_list = _vtk.vtkIdList()
    locator.FindCellsWithinBounds(list(bounds), id_list)
    return vtk_id_list_to_array(id_list)


def unitVector(vector):
    vector = np.array(vector)
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    numpy.clip(a, a_min, a_max, out=None, **kwargs)[source]
    Clip (limit) the values in an array.     Given an interval, values outside the interval
    are clipped to the interval edges. For example, if an interval of [0, 1] is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.
    Equivalent to but faster than np.minimum(a_max, np.maximum(a, a_min)).
    No check is performed to ensure a_min < a_max.
    """
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


if "__name__" == "__main__":
    bbox = np.array((0.0, 12.21, 0.5, 10.4, 0.20, 5.37))
    voxel_size = [0.11, 0.11, 0.044]

    mesh_background = background_mesh_generator(bbox, voxel_size)

    # mesh_background.save('./test_bbox_voxel.vtu', binary=True)