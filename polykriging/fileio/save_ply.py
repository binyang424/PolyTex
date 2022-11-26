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

    Examples
    --------
    >>> import numpy as np
    >>> import polykriging as pk
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> cells = [("triangle", [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])]
    >>> point_data = {"a": np.array([0, 1, 2, 3])}
    >>> cell_data = {"b": np.array([0, 1, 2, 3])}
    >>> pk.save_ply("test.ply", vertices, cells, point_data, cell_data)
    >>> print("Done")
    Done

    """
    mesh = meshio.Mesh(points=vertices,
                       cells=cells,
                       # Optionally provide extra data on points, cells, etc.
                       point_data=point_data,
                       # Each item in cell data must match the cells array
                       cell_data=cell_data,
                       )
    # check if the extension in file is in [ply, stl, vtk, vtu]
    import os
    filename, file_extension = os.path.splitext(file)
    if file_extension in [".ply", ".stl", ".vtk", ".vtu"]:
        meshio.write(file, mesh, binary=binary)
        print("The mesh is saved as " + filename+file_extension + " file successfully.")
    else:
        raise ValueError("The file extension is not supported. "
                         "Please use ply, stl, vtk, or vtu.")

    return None


def get_ply_property(mesh_path, column, skip=11, type="vertex", save_vtk=False):
    """
    This function get a vertex property or cell property from a mesh stored as .ply
    format. It is intended to be used to get the user-defined properties that most of
    meshing and rendering software does not support.

    Note
    ----
    The mesh must be saved as ASCII format.

    Parameters
    ----------
    mesh_path : str
        The path of the mesh file with .ply extension.
    column : int or list of int
        The column number of the property.
    skip : int, optional
        The number of lines to skip in the header. The default is 11.
    type : str, optional
        The type of the property. The default is "vertex" for vertex property. The other
        possible value is "cell" for cell property.
    save_vtk : bool, optional
        If True, the mesh is saved as a vtk file. The default is False.

    Returns
    -------
    property : numpy.ndarray
        The property of the mesh.

    Examples
    --------
    >>> import polykriging as pk
    >>> mesh_path = "./weft_0_lin_lin_krig_30pts.ply"
    >>> quality = pk.get_ply_property(mesh_path, -2, skip=11, type="vertex", save_vtk=False)
    >>> quality
    """
    import pyvista as pv
    import numpy as np

    mesh = pv.read(mesh_path)
    n_pts = mesh.n_points
    n_cells = mesh.n_cells

    # load as csv
    mesh_txt = np.loadtxt(mesh_path, dtype=object, delimiter=" ", skiprows=11)

    if type == "vertex":
        vertex = mesh_txt[:n_pts]
        quality = vertex[:, column].astype(np.float32)
    elif type == "cell":
        cell = mesh_txt[n_pts:n_pts + n_cells]
        quality = cell[:, column].astype(np.float32)

    if save_vtk:
        mesh["quality"] = quality
        mesh.save(mesh_path[:-4] + ".vtk")

    return quality


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # in terminal: python3 save_ply.py -v
