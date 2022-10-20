import numpy as np
import pandas as pd

from numpy.compat import os_fspath
from numpy.lib import format
import contextlib
from ..kriging.curve2D import addPoints
from ..thirdparty.bcolors import bcolors
from .save_krig import save_krig
import os


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


def pk_save(file, df):
    """
    Save a Python dict or pandas dataframe as a file format defined in polykriging (.coo, geo) file

    Parameters
    ----------
    file: str, or pathlib.Path.
        File path and name to which the data is saved.
    df: pandas.DataFrame or dict
        The data to be saved.

    Returns
    -------
    None
    """
    # df is a dict
    filename = os.path.basename(file)
    if file.endswith('.krig'):
        expr = save_krig(df)
        expr.save(file)
        print(bcolors.ok("The Kriging function {} is saved successfully.").format(filename))

    elif isinstance(df, dict) and not file.endswith('.krig'):
        save(file, df, allow_pickle=True)
        print(bcolors.ok("The file {} is saved successfully.").format(filename))

    elif isinstance(df, pd.DataFrame):
        # index
        index = df.index
        # columns
        columns = df.columns
        # values
        values = df.to_numpy()
        pk_file = {"index": index, "columns": columns, "values": values}
        save(file, pk_file, allow_pickle=True, fix_imports=True)
        print(bcolors.ok("The file {} is saved successfully.").format(filename))
    else:
        raise TypeError("The input data type is not supported.")


def pk_load(pk_file):
    """
    Load a file format defined in polykriging (.coo, .geo, or .stat) file
    and return as a pandas dataframe or a numpy.array object.

    Parameters
    ----------
    pk_file:  str, or pathlib.Path.
        File path and name to which the data is stored.

    Returns
    -------
    df: pandas.DataFrame or numpy.ndarray
        The data to be loaded. It is a pandas dataframe if the file is a .coo/geo file.
        Otherwise, it is a numpy array or dict and a warning will be raised.
    """

    filename = os.path.basename(pk_file)
    if pk_file.endswith('.krig'):
        print(bcolors.ok("The Kriging expression {} is loaded successfully.").format(filename))
        return save_krig.load(pk_file)
    try:
        file = np.load(pk_file, allow_pickle=True, fix_imports=True).tolist()
        index = file["index"]
        columns = file["columns"]
        values = file["values"]

        df = pd.DataFrame(values, index=index, columns=columns)
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))
    except:
        df = np.load(pk_file, allow_pickle=True, fix_imports=True)
        print(bcolors.ok("The file {} is loaded successfully.").format(filename))
        print(bcolors.warning("Warning: The file is not a pandas dataframe. It is loaded as a numpy array.\n "
              "If it is a dict originally, please use file.tolist() to convert it to a dict."))
    return df


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


def coo_to_ply(file_coo, file_ply, binary=False, interpolate=False, threshold=0.1):
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


if "__main__" == __name__:
    label_row = pd.date_range("20130101", periods=6, freq="D", tz="UTC")
    label_col = list("ABCD")
    data = np.random.randn(6, 4)
    df = pd.DataFrame(data, index=label_row, columns=label_col)

    # save
    pk_save("test.coo", df)

    # load
    df = pk_load("test.coo")
