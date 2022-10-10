# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from pyvista import _vtk


def isInbBox(bbox, point):
    """
    Determine if point is within a bounding box (bbox)
    that parallel to the principle axes of global Cartesian coordinate.

    Parameters
    ----------
    bbox: list
        bounding box, [xmin, xmax, ymin, ymax, zmin, zmax].
    point: list
        [x, y, z]

    Returns
    -------
    True or False
    """
    x, y, z = point[0], point[1], point[2]
    xmin, ymin, zmin = bbox[0], bbox[1], bbox[2]
    xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]
    boolst = [True, True, True, False, False, False]
    boo = [x > xmin, y > ymin, z > zmin, x > xmax, y > ymax, z > zmax]
    return boo == boolst


def background_mesh_generator(bbox, voxel_size=None):
    """
    Generate a voxel background mesh.

    Parameters
    ----------
    bbox: bounding box of the background mesh specified through a numpy array
        contains the minimum and maximum coordinates of the bounding box
        [xmin, xmax, ymin, ymax, zmin, zmax]
    voxel_size: voxel size of the background mesh, type: None, float, or numpy.ndarray
        if None, the voxel size is set to the 1/20 of the diagonal length of the bounding box;
        if float, the voxel size is set to the float value in x, y, z directions;
        if numpy.ndarray, the voxel size is set to the values in the numpy.ndarray for corresponding directions.

    Returns
    -------
    grid : pyvista mesh object (UnstructuredGrid)
    mesh_shape : tuple
        shape of the mesh
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
    mesh_shape = np.array(x.shape) - 1
    print(mesh_shape)
    # return the unstructured grid and the number of cells of the grid in each direction
    return grid.cast_to_unstructured_grid(), mesh_shape


def find_cells_within_bounds(mesh, bounds):
    """
    Find the index of cells in this mesh within bounds.

    Parameters
    ----------
    mesh: pyvista mesh
    bounds: type:　iterable(float)
        list of 6 values, [xmin, xmax, ymin, ymax, zmin, zmax]

    Returns
    -------
    type: numpy.ndarray
        array of cell indices within bounds.
    
    Example
    -------
        >> mesh = pv.PolyData(np.random.rand(10, 3))
        >> indices = find_cells_within_bounds(mesh, [0, 1, 0, 1, 0, 1])
    """
    from pyvista import _vtk
    from pyvista import vtk_id_list_to_array

    if np.array(bounds).size != 6:
        raise TypeError("Bounds must be a length three tuple of floats.")
    locator = _vtk.vtkCellTreeLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    id_list = _vtk.vtkIdList()
    locator.FindCellsWithinBounds(list(bounds), id_list)
    return vtk_id_list_to_array(id_list)


def label_mask(mesh_background, mesh_tri, tolerance=0.0000001):
    """
    Store the label of each fiber tow for intersection detection.

    Parameters
    ----------
    mesh_background: pyvista.UnstructuredGrid
        background mesh
    mesh_tri: pyvista.PolyData
        tubular mesh of the fiber tows
    tolerance: float
        tolerance for the enclosed point detection

    Returns
    -------
    mask: type: numpy.ndarray (bool)
        mask of the background mesh, True for the cells that are within the bounds of the tubular mesh
    label_yarn: type: numpy.ndarray (int) (1D)
    """
    # extract the cell centers of the background mesh
    cellCenters = mesh_background.cell_centers().points
    points_poly = pv.PolyData(cellCenters)
    # find the cells that are within the tubular surface of the fiber tow
    select = points_poly.select_enclosed_points(mesh_tri, tolerance=0.0000001, check_surface=True)
    mask = select['SelectedPoints'] == 1
    label_yarn = select['SelectedPoints']
    return mask, label_yarn


def intersection_detect(label_set_dict):
    """
    Find the intersection of fiber tows from implicit surface.

    Parameters
    ----------
    label_set_dict: dictioanry
        dictionary of the label sets of the fiber tows
        (key: yarn indices, value: sparse matrix of cell queries)

    Returns
    -------
    type: dictionary of the indices of intersected cell
        key: yarn indices 1_yarn indices 2, value: sparse matrix of cell indices
    """
    import itertools
    from scipy.sparse import csr_matrix, coo_matrix

    # length and keys of the dictionary
    length = len(label_set_dict)
    keys = label_set_dict.keys()
    # possible intersection of cell indices
    tows_combination_list = list(itertools.combinations(keys, 2))

    print(tows_combination_list)

    intersect_info = []
    intersect_info_dict = {}
    for i, key in enumerate(tows_combination_list):
        # print the intersection checking progress
        # print("Intersection check of Yarn %d and %d " % (key[0], key[1]))
        # find the intersection of 2 yarns
        intersection = label_set_dict[key[0]] + label_set_dict[key[1]]
        # check if any element in the sparse matrix intersection is larger than 1
        if np.any(intersection.data > 1):
            print("Intersection found between Yarn %d and %d" % (key[0], key[1]))
            n_cells_yarn1 = label_set_dict[key[0]].data.size
            n_cells_yarn2 = label_set_dict[key[1]].data.size
            n_cells_intersection = np.sum(intersection.data > 1)
            # find the ratio of intersected cells to the total cells in each yarn
            ratio_yarn1 = n_cells_intersection / n_cells_yarn1 * 100
            ratio_yarn2 = n_cells_intersection / n_cells_yarn2 * 100
            # print the ratio of intersected cells to the total cells in each yarn
            print("Ratio of intersected cells to the total cells in each yarn: \n"
                  " %.2f%% ( %d / %d) and %.2f%% (%d / %d)" % (ratio_yarn1, n_cells_intersection,
                                                               n_cells_yarn1, ratio_yarn2, n_cells_intersection,
                                                               n_cells_yarn2))
            # print a dash line to separate the information
            print("-" * 50)
            # [yarn1, yarn2, intersected cells, cells in yarn 1, ratio of intersected cells in yarn 1,
            # cells in yarn2, ratio of intersected cells in yarn 2]
            intersect_info.append([key[0], key[1], n_cells_intersection,
                                   n_cells_yarn1, ratio_yarn1, n_cells_yarn2, ratio_yarn2])
            # save the intersection information in a dictionary
            intersect_info_dict[key] = intersection

    # summarize all the labels in label_set_dict
    keys = label_set_dict.keys()
    for key in keys:
        try:
            cell_data_intersect += label_set_dict[key]
        except NameError:
            cell_data_intersect = label_set_dict[key]

    return intersect_info, intersect_info_dict, np.array(cell_data_intersect.todense())[0]


def structured_cylinder_vertices(a, b, h, theta_res=5, h_res=5):
    """
    Generate points on an ellipse.

    Parameters
    ----------
    a : float
        semi-major axis
    b : float
        semi-minor axis
    h : float
        height
    theta_res : int, optional
        number of points, by default 5.
    h_res:  int, optional
        number of points. by default 5.

    Returns
    -------
    points: numpy.ndarray
        vertices on the ellipse surface (x, y, z).
    """
    theta_resolution = theta_res + 1
    theta = np.linspace(0, 2 * np.pi, theta_resolution)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    z = np.linspace(0, h, h_res, endpoint=True)

    points = np.zeros((theta_resolution * h_res, 3))
    for i in np.arange(h_res):
        points[theta_resolution * i:theta_resolution * (i + 1), :2] = np.vstack((x, y)).T
        points[theta_resolution * i:theta_resolution * (i + 1), 2] = z[i]

    return points


def tubular_mesh_generator(theta_res, h_res, vertices, plot=True):
    """
    Generate a tubular mesh.

    Parameters
    ----------
    theta_res: int
        number of points
    h_res: int
        number of points
    vertices: numpy.ndarray
        vertices of the tubular mesh, shape (n, 3)
        The vertices of the tubular mesh are sorted in the radial direction first,
        then in the vertical direction. The first vertex is repeated
        at the end of each radial direction point list.

    Returns
    -------
    mesh : points on the tubular mesh
    """
    import pyvista as pv
    mesh = pv.CylinderStructured(theta_resolution=theta_res + 1,
                                 z_resolution=h_res)
    mesh.points = vertices
    if plot:
        mesh.plot(show_edges=True)
    return mesh


def to_meshio_data(mesh, theta_res, correction=True):
    """
    Convert PyVista flavor data structure to meshio.

    Parameters
    ----------

    mesh: PyVista.DataSet
        Any PyVista mesh/spatial data type.
    theta_res:
        number of points in the radial direction
    correction: boolean
        if True, tubular mesh will be closed at the ends with triangles.
    """
    try:
        import meshio
    except ImportError:  # pragma: no cover
        raise ImportError("To use this feature install meshio with:\n\npip install meshio")

    try:  # for meshio<5.0 compatibility
        from meshio.vtk._vtk import vtk_to_meshio_type
    except:  # noqa: E722 pragma: no cover
        from meshio._vtk_common import vtk_to_meshio_type

    # Cast to pyvista.UnstructuredGrid
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()

    # Copy useful arrays to avoid repeated calls to properties
    vtk_offset = mesh.offset
    vtk_cells = mesh.cells
    vtk_cell_type = mesh.celltypes

    # Check that meshio supports all cell types in input mesh
    pixel_voxel = {8, 11}  # Handle pixels and voxels
    for cell_type in np.unique(vtk_cell_type):
        if cell_type not in vtk_to_meshio_type.keys() and cell_type not in pixel_voxel:
            raise TypeError(f"meshio does not support VTK type {cell_type}.")

    # Get cells
    cells = []
    c = 0
    for offset, cell_type in zip(vtk_offset, vtk_cell_type):
        numnodes = vtk_cells[offset + c]
        if _vtk.VTK9:  # must offset by cell count
            cell = vtk_cells[offset + 1 + c: offset + 1 + c + numnodes]
            c += 1
        else:
            cell = vtk_cells[offset + 1: offset + 1 + numnodes]
        cell = (
            cell
            if cell_type not in pixel_voxel
            else cell[[0, 1, 3, 2]]
            if cell_type == 8
            else cell[[0, 1, 3, 2, 4, 5, 7, 6]]
        )
        cell_type = cell_type if cell_type not in pixel_voxel else cell_type + 1
        cell_type = vtk_to_meshio_type[cell_type] if cell_type != 7 else f"polygon{numnodes}"

        if len(cells) > 0 and cells[-1][0] == cell_type:
            cells[-1][1].append(cell)
        else:
            cells.append((cell_type, [cell]))

    for k, c in enumerate(cells):
        cells[k] = (c[0], np.array(c[1]))

    # Get point data
    point_data = {k.replace(" ", "_"): v for k, v in mesh.point_data.items()}

    # Get cell data
    vtk_cell_data = mesh.cell_data
    n_cells = np.cumsum([len(c[1]) for c in cells[:-1]])
    cell_data = (
        {k.replace(" ", "_"): np.split(v, n_cells) for k, v in vtk_cell_data.items()}
        if vtk_cell_data
        else {}
    )

    points = np.array(mesh.points)
    cells = cells
    if correction:
        points, cells = mesh_correction(cells, points, theta_res=theta_res)

    point_data = np.array(point_data)
    cell_data = cell_data

    return points, cells, point_data, cell_data


def mesh_correction(cells, points, theta_res):
    cells_connectivity = cells[0][1]

    # 遍历cells中的每一个元素，如果元素中的点在rm_row_ind中，则将该元素减小theta_res
    rm_row_ind = np.arange(theta_res, points.shape[0] + 2, theta_res + 1)
    for i in range(cells_connectivity.shape[0]):
        for j in range(cells_connectivity.shape[1]):
            if cells_connectivity[i][j] in rm_row_ind:
                # print(cells_connectivity[i][j])
                cells_connectivity[i][j] -= theta_res

    # get index of boundary points
    first_boundary_ind = np.arange(theta_res)
    second_boundary_ind = np.arange(points.shape[0] - 1 - theta_res, points.shape[0] - 1)

    # coordinates of boundary points
    first_boundary = points[first_boundary_ind]
    second_boundary = points[second_boundary_ind]

    # calculate the centroids of the boundaries
    centroid_first_boundary = np.mean(first_boundary, axis=0)
    centroid_second_boundary = np.mean(second_boundary, axis=0)

    # add the centroids to the points
    points = np.vstack((points, centroid_first_boundary, centroid_second_boundary))

    # new cells
    first_boundary_new_cell = [np.array([points.shape[0] - 2, i + 1, i]) for i in first_boundary_ind[:-1]]
    first_boundary_new_cell.append(np.array([points.shape[0] - 2, first_boundary_ind[0], first_boundary_ind[-1]]))

    second_boundary_new_cell = [np.array([points.shape[0] - 1, i, i + 1]) for i in second_boundary_ind[:-1]]
    second_boundary_new_cell.append(np.array([points.shape[0] - 1, second_boundary_ind[-1], second_boundary_ind[0]]))
    new_cells = first_boundary_new_cell + second_boundary_new_cell

    cells = [('quad', list(cells_connectivity)), ('triangle', new_cells)]

    return points, cells


def unit_vector(vector):
    """
    Returns the unit vector of the input vector.

    Parameters
    ----------
    vector : array-like
        Input vector.

    Returns
    -------
    unit_vector : array-like
        Unit vector of the input vector.
    """
    vector = np.array(vector)
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    numpy.clip(a, a_min, a_max, out=None, **kwargs)[source]
    Clip (limit) the values in an array.     Given an interval, values outside the interval
    are clipped to the interval edges. For example, if an interval of [0, 1] is specified,
    values smaller than 0 become 0, and values larger than 1 become 1.
    Equivalent to but faster than np.minimum(a_max, np.maximum(a, a_min)).
    No check is performed to ensure a_min < a_max.

    Parameters
    ----------
    v1 : array-like
        First vector.
    v2 : array-like
        Second vector.

    Returns
    -------
    angle : float
        Angle in degrees between the two input vectors.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


if "__name__" == "__main__":
    bbox = np.array((0.0, 12.21, 0.5, 10.4, 0.20, 5.37))
    voxel_size = [0.11, 0.11, 0.044]

    mesh_background = background_mesh_generator(bbox, voxel_size)

    # mesh_background.save('./test_bbox_voxel.vtu', binary=True)
