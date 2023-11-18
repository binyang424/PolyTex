# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import os, re

import pickle
import pandas as pd
import pyvista as pv
import sympy
import vtk

import numpy as np
from numpy.compat import os_fspath
from numpy.lib import format
from tkinter import Tk, filedialog, messagebox

from .kriging.curve2D import addPoints
from .thirdparty.bcolors import bcolors


def construct_tetra_vtk(points, cells, save=None, binary=True):
    """
    Construct a UnstructuredGrid tetrahedral mesh from vertices and connectivity.

    Parameters
    ----------
    points: (n, 3) array
        vertices
    cells: (m, 4) array
        connectivity
    save: str
        The path and file name of the vtk file to be saved ("./tetra.vtk").
        If None, the vtk file will not be saved.
    binary: bool
        whether to save the mesh in binary format

    Returns
    -------
    grid: pyvista.UnstructuredGrid
        UnstructuredGrid tetrahedral mesh
    """
    n_cells = cells.shape[0]
    offset = np.array([4 * i for i in np.arange(n_cells)])
    cells = np.concatenate(np.insert(cells, 0, 4, axis=1)).astype(np.int64)
    cell_type = np.array([vtk.VTK_TETRA] * n_cells)
    grid = pv.UnstructuredGrid(offset, cells, cell_type, np.array(points))
    if save is not None:
        grid.save(save, binary=binary)
    return grid


def cwd_chdir(path=""):
    """
    Set given directory or the folder where the code file is as current working directory

    Parameters
    ----------
    path:
        the path of current working directory. if empty, the path of the code file is used.

    Returns
    -------
    cwd: the current working directory.
    """
    import sys

    if path == "":
        cwd = str(sys.argv[0])
        cwd, pyname = os.path.split(cwd)
    else:
        cwd = path
    os.chdir(cwd)
    return cwd


def choose_directory(titl='Select the target directory:'):
    """
    Choose a directory with GUI and return its path.

    Parameters
    ----------
    titl: String.
        The title of the open folder dialog window.

    Returns
    -------
    path: String.
        The path of the selected directory.
    """

    print(titl)
    # pointing root to Tk() to use it as Tk() in program.
    # like a window (container) where we can put widgets.
    directory_root = Tk()
    directory_root.withdraw()  # Hides small tkinter window.
    directory_root.attributes('-topmost',
                              True)  # Opened windows will be active. above all windows.
    path_work = filedialog.askdirectory(title=titl)  # Returns opened path as str
    if path_work == '':
        top = Tk()
        top.withdraw()
        top.geometry("150x150")
        message = messagebox.askquestion("Warning", "You did not select any folder! "
                                                    "Do you wish to select again?")
        if message == 'yes':
            return choose_directory()
        elif message == 'no':
            return None
    else:
        # replace the forward slash returned by askdirectory
        # with backslash (\) on Windows.
        return path_work.replace('/', os.sep)


def filenames(path, filter="csv"):
    """
    Get the list of files in the given folder.

    Parameters
    ----------
    path:
        the path of the folder
    filter:
        filter for file selection.++

    Returns
    -------
    flst: the list of files in the given folder.
    """
    filenamels = os.listdir(path)
    # filter the file list by the given filter.
    flst = [x for x in filenamels if (filter in x)]
    flst.sort()
    return flst


def zip_files(directory, file_list, filename, remove="True"):
    """
    Add multiple files to the zip file.

    Parameters
    ----------
    directory: String.
        The directory of the files to be added to zip file. Therefore,
        all the files in the file_list should be in the same directory.
    file_list : List.
        The list of file names to be added to the zip file (without directory).
    filename: String.
        The name of the zip file. The zip file is saved in the same directory
    remove:
        Whether to remove original files after adding to zip file.
        Default is True. If False, the original files will not be removed.

    Returns
    -------
    None.
    """
    from zipfile import ZipFile

    # check extension of the zip file
    if filename[-4:] != ".zip":
        filename += ".zip"

    with ZipFile(filename, 'w') as zipObj:
        for i in range(len(file_list)):
            zipObj.write(directory + file_list[i])

            if remove == "True":
                os.remove(directory + file_list[i])
    print(bcolors.ok("Zip file saved as " + filename))


def choose_file(titl='Select the target directory:', format='csv'):
    """
    Choose a file with GUI and return its path.

    Parameters
    ----------
    titl: String.
        The title of the window.

    Returns
    -------
    path: String.
        The path of the file.
    """

    print(titl)
    directory_root = Tk()
    directory_root.withdraw()  # Hides small tkinter window.
    directory_root.attributes('-topmost',
                              True)  # Opened windows will be active (appears above all windows)
    path_work = filedialog.askopenfilename(
        title=titl, filetypes=[(format, format), ('All files', '*.*')])  # Returns opened path as str

    # replace the forward slash returned by askdirectory
    # with backslash (\) on Windows.
    return path_work.replace('/', os.sep)


def save_nrrd(cell_label, file_name, file_path='./'):
    """
    Save the labels of a hexahedral mesh to a nrrd file. The labels should be
    starting from 0 and increasing by 1.

    Parameters
    ----------
    cell_label: numpy array(int, int, int)
        The cell label of the mesh.
    file_name: String
        The name of the .nrrd file.
    file_path: String
        The save path of the .nrrd file.

    Returns
    -------
    None
    """
    import nrrd

    indicator = np.zeros_like(cell_label)
    for i, label in enumerate(np.unique(cell_label)):
        mask = cell_label == label
        indicator[mask] = i

    header = {'space origin': [0, 0, 0],
              "space directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
              'space': 'left-anterior-superior'}

    # Write to a NRRD file with pynrrd
    if file_name[-5:] != '.nrrd':
        file_name += '.nrrd'
    nrrd.write(file_path + file_name, indicator, header, index_order='C')
    return indicator


class save_krig(dict):
    """
    This class saves a dictonary of sympy expressions to a file in human
    readable form and then load as sympy expressions directly without other
    conversion. It is called by polykriging.fileio.pk_save to save kriging
    expressions to a ".krig" file and by polykriging.fileio.pk_load to load
    these files. Therefore, the class is not intended to be used directly by
    the user.

    Note:
    --------
    This class is taken from: https://github.com/sympy/sympy/issues/7974.
    A bug in exec() is fixed and some modifications are made to make it
    fit for the purpose of this project (store the kriging expression).

    Example:
    --------
    >>> import sympy
    >>> from polykriging.fileio.save_krig import save_krig
    >>> a, b = sympy.symbols('a, b')
    >>> d = save_krig({'a':a, 'b':b})
    >>> d.save('name.krig')
    >>> del d
    >>> d2 = save_krig.load('name.krig')
    """

    def __init__(self, *args, **kwargs):
        super(save_krig, self).__init__(*args, **kwargs)

    def __repr__(self):
        d = dict(self)
        for key in d.keys():
            d[key] = sympy.srepr(d[key])
        # regex is just used here to insert a new line after
        # each dict key, value pair to make it more readable
        return re.sub('(: \"[^"]*\",)', r'\1\n', d.__repr__())

    def save(self, file):
        with open(file, 'w') as savefile:
            savefile.write(self.__repr__())

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as loadfile:
            # Note that the variable name temp should not be the same as the other
            # local variables in the function, otherwise exec will not work and will
            # raise an NameError: name 'temp' is not defined.
            exec("temp =" + loadfile.read())
        # convert the strings back to sympy expressions and return a new save_krig.
        # This is done by calling the save_krig constructor with the new dict.
        # locals() is used to get the sympy symbols from the exec statement above.
        d = locals()['temp']
        for key in d.keys():
            d[key] = sympy.sympify(d[key])
        return cls(d)


def save(file, arr, allow_pickle=True, fix_imports=True):
    """
    This is an exact copy of numpy.save, except that it does not check the extensions.

    Parameters
    ----------
    file : file, str, or pathlib.Path. File or filename to which the data is saved.
    arr : array_like. Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for disallowing
        pickles include security (loading pickled data can execute arbitrary
        code) and portability (pickled objects may not be loadable on different
        Python installations, for example if the stored objects require libraries
        that are not available, and not all pickled data is compatible between
        Python 2 and Python 3).
        Default: True
    fix_imports : bool, optional
        Only useful in forcing objects in object arrays on Python 3 to be
        pickled in a Python 2 compatible way. If `fix_imports` is True, pickle
        will try to map the new Python 3 names to the old module names used in
        Python 2, so that the pickle data stream is readable with Python 2.
    """

    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os_fspath(file)
        # if not file.endswith('.npy'):
        #     file = file + '.npy'
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle,
                           pickle_kwargs=dict(fix_imports=fix_imports))


def pk_save(fp, data):
    """
    Save a Python dict or pandas dataframe as a file format defined in polykriging (.coo, geo) file

    Parameters
    ----------
    fp: str
        File path and name to which the data is saved. If the file name does not end with
        a supported file extension, a ValueError will be raised.
    data: Tow, Tex, or dict
        The data to be saved. It can be several customised file formats for polykriging.


    Returns
    -------
    None
    """
    filename = os.path.basename(fp)
    # get file extension
    ext = os.path.splitext(filename)[1]

    if ext == "":
        raise ValueError("The file extension is not given. Supported file extensions are "
                         ".coo, .geo, .tow, .tex, and .krig.")
    elif ext not in ['.coo', '.geo', '.tow', '.tex', '.krig']:
        raise ValueError("The file extension is not supported. Supported file extensions are "
                         ".coo, .geo, .tow, .tex, and .krig.")

    if fp.endswith('.krig'):
        expr = save_krig(data)
        expr.save(fp)
        print(bcolors.ok("The Kriging function {} is saved successfully.").format(filename))
    elif ext in ['.tow', '.tex']:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        f.close()
        print(bcolors.ok("The file {} is saved successfully.").format(filename))
    elif isinstance(data, pd.DataFrame):  # save as .coo or .geo file
        data.to_pickle(fp)
        print(bcolors.ok("The file {} is saved successfully.").format(filename))
    else:
        raise TypeError("The input data type is not supported.")


def pk_load(file):
    """
    Load a file format defined in polykriging (.coo, .geo, or .stat) file
    and return as a pandas dataframe or a numpy.array object.

    Parameters
    ----------
    file:  str, or pathlib.Path.
        File path and name to which the data is stored.

    Returns
    -------
    df: pandas.DataFrame or numpy.ndarray
        The data to be loaded. It is a pandas dataframe if the file is a .coo/geo file.
        Otherwise, it is a numpy array or dict and a warning will be raised.
    """

    filename = os.path.basename(file)
    ext = os.path.splitext(filename)[1]

    if ext == "":
        raise ValueError("The file extension is not given. Supported file extensions are "
                         ".pcd, .coo, .geo, .tow, .tex, and .krig.")
    elif ext not in ['.pcd', '.coo', '.geo', '.tow', '.tex', '.krig']:
        raise ValueError("The file extension is not supported. Supported file extensions are "
                         ".pcd, .coo, .geo, .tow, .tex, and .krig.")

    if file.endswith('.krig'):
        print(bcolors.ok("The Kriging expression {} is loaded successfully.").format(filename))
        return save_krig.load(file)

    if ext in ['.coo', '.geo']:
        data = pd.read_pickle(file)
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))
    elif ext in ['.tow', '.tex']:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        f.close()
    else:
        data = np.load(file, allow_pickle=True, fix_imports=True).tolist()
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))

    return data


def pcd_to_ply(file_pcd, file_ply, binary=False):
    """
    Convert a pcd file to ply file.

    Parameters
    ----------
    file_pcd : str
        The path of the pcd file or pathlib.Path. File or filename to which the data is saved.
    file_ply : str
        The path of the ply file or pathlib.Path. File or filename to which the data is to be saved.
    :return: None
    """
    import meshio

    vertices = pk_load(file_pcd).to_numpy()

    mesh = meshio.Mesh(points=vertices[:, 1:],
                       cells=[],
                       # Optionally provide extra data on points, cells, etc.
                       point_data={"nx": vertices[:, 0], "ny": vertices[:, 1], "nz": vertices[:, 2]},
                       # Each item in cell data must match the cells array
                       # cell_data={"a": [[0.1, 0.2], [0.4]]},
                       )

    meshio.write(file_ply, mesh, binary=False)


def coo_to_ply(file_coo, file_ply, interpolate=False, threshold=0.1):
    """
    Convert a pcd file to ply file.

    Parameters
    ----------
    file_coo : str
        The path of the coo file or pathlib.Path. File or filename to which the data is saved.
    file_ply : str
        The path of the ply file or pathlib.Path. File or filename to which the data is to be saved.
    :return: None
    """
    import meshio
    df = pk_load(file_coo)
    vertices = df.to_numpy()[:, [1, 3, 4, 5]]

    if interpolate:
        # interpolate
        vertices = addPoints(vertices, threshold=threshold)

    mesh = meshio.Mesh(points=vertices[:, 1:],
                       cells=[],
                       # Optionally provide extra data on points, cells, etc.
                       point_data={"nx": vertices[:, 0], "ny": vertices[:, 1], "nz": vertices[:, 2]},
                       # Each item in cell data must match the cells array
                       # cell_data={"a": [[0.1, 0.2], [0.4]]},
                       )

    meshio.write(file_ply, mesh, binary=False)


def save_ply(file, vertices, cells=[], point_data={}, cell_data={}, binary=False):
    print(bcolors.warning(
        "This function will be deprecated in the future. Please use polykriging.meshio_save() instead."))
    return meshio_save(file, vertices, cells, point_data, cell_data, binary)


def meshio_save(file, vertices, cells=[], point_data={}, cell_data={}, binary=False):
    """
    Save surface mesh as a mesh file by definition of vertices and faces. Point data and cell data can be added.
    It is a wrapper of meshio.write() function.

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
        The cell data of the mesh. The default is {}. Note that the cell data should be added as a
        list of arrays. Each array in the list corresponds to a cell type. For example, if the mesh
        has 2 triangles and 1 quad, namely,
          cells = [("triangle", [0, 1, 2], [1,2,3]), ("quad", [3, 4, 5, 6])],
        then the cell data should be added as
          cell_data = {"data": [[1, 2], [3]}.
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
    >>> cell_data = {"b": np.array([[0, 1, 2, 3],])}
    >>> pk.meshio_save("test.ply", vertices, cells, point_data, cell_data)
    >>> print("Done")
    Done

    """
    try:
        import meshio
    except ModuleNotFoundError:
        raise ImportError("This function requires meshio package but it is not installed.")

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
        print("The mesh is saved as " + filename + file_extension + " file successfully.")
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


def save_csv(filename, dataset, csv_head):
    """
    Save numpy array to csv file with given info in the first row.

    Parameters
    ----------
    filename:
        The path and name of the csv file.
    dataset: List or numpy.ndarray
        The dataset to be saved in the csv file
    csv_head:
        A list of headers of the csv file. The length of the list
        should be the same as the number of columns in the dataset.

    Returns
    -------
    None.

    """
    import csv

    if filename[-4:] != ".csv":
        filename = filename + ".csv"

    path = filename + ".csv"

    with open(path, 'w', newline="") as f:
        csv_write = csv.writer(f)

        csv_write.writerow(csv_head)
        for row in dataset:
            csv_write.writerow(row)
    return 1


if "__main__" == __name__:
    import doctest

    doctest.testmod()
