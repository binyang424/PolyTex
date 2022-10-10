import numpy as np
import meshio


def save_ply(file, vertices, cells=[], point_data={}, cell_data={}, binary=False):
    """
    Save surface mesh as a ply file by definition of vertices and faces. Point data and cell data can be added.

    Parameters
    ----------
    file : str
        The path of the ply file or pathlib.Path. File or filename to which the data is saved.
    vertices : numpy.ndarray
        The vertices of the mesh. The shape of the array is (n, 3), where n is the number of vertices.
    cells : list, optional
        The faces of the mesh stored as the connectivity between vertices. The default is [].
    point_data : dict, optional
        The point data of the mesh. The default is {}.
    cell_data : dict, optional
        The cell data of the mesh. The default is {}.
    binary : bool, optional
        If True, the data is written in binary format. The default is False.

    Returns
    -------
    None.
    """
    mesh = meshio.Mesh(points=vertices,
                       cells=[],
                       # Optionally provide extra data on points, cells, etc.
                       point_data={"nx": vertices[:, 0], "ny": vertices[:, 1], "nz": vertices[:, 2]},
                       # Each item in cell data must match the cells array
                       # cell_data={"a": [[0.1, 0.2], [0.4]]},
                       )

    meshio.write(file, mesh, binary=False)

    return None
