import numpy as np
import pandas as pd

from numpy.compat import os_fspath
from numpy.lib import format
import contextlib


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
    Save a pandas dataframe as a file format defined in polykriging (.coo, geo) file
    :param file: file, str, or pathlib.Path. File or filename to which the data is saved.
    :param df: pandas.DataFrame
    :return: None
    """
    # index
    index = df.index
    # columns
    columns = df.columns
    # values
    values = df.to_numpy()

    pk_file = {"index": index, "columns": columns, "values": values}
    # save dataframes to npz file
    save(file, pk_file, allow_pickle=True, fix_imports=True)


def pk_load(pk_file):
    """
    Load a file format defined in polykriging (.coo, geo) file
    and return as a pandas dataframe.
    :param pk_file:  file, str, or pathlib.Path. File or filename to which the data is stored.
    :return df: pandas.DataFrame
    """
    file = np.load(pk_file, allow_pickle=True, fix_imports=True).tolist()
    index = file["index"]
    columns = file["columns"]
    values = file["values"]

    df = pd.DataFrame(values, index=index, columns=columns)
    return df


if "__main__" == __name__:
    label_row = pd.date_range("20130101", periods=6, freq="D", tz="UTC")
    label_col = list("ABCD")
    data = np.random.randn(6, 4)
    df = pd.DataFrame(data, index=label_row, columns=label_col)

    # save
    pk_save("test.coo", df)

    # load
    df = pk_load("test.coo")