import pyvista as pv
import numpy as np
import vtk


def construct_tetra_vtk(points, cells, save=False, filename="tetra.vtk", path="./", binary=True):
    """
    Construct a UnstructuredGrid tetrahedral mesh from vertices and connectivity.
    :param points: (n, 3) array
        vertices
    :param cells: (m, 4) array
        connectivity
    :param save: bool
        whether to save the mesh
    :param filename: str
        if save=True, provide a file name
    :param path: str
        if save=True, provide a path to save the mesh
    :param binary: bool
        whether to save the mesh in binary format
    :return grid: pyvista.UnstructuredGrid
        UnstructuredGrid tetrahedral mesh
    """
    n_cells = cells.shape[0]
    offset = np.array([4 * i for i in np.arange(n_cells)])
    cells = np.concatenate(np.insert(cells, 0, 4, axis=1)).astype(np.int64)
    cell_type = np.array([vtk.VTK_TETRA] * n_cells)
    grid = pv.UnstructuredGrid(offset, cells, cell_type, np.array(points))
    if save:
        grid.save(path + filename, binary=binary)
    return grid