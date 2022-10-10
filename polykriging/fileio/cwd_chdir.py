# ！/usr/bin/env python
# -*- coding: utf-8 -*-


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
    import sys, os
    if path == "":
        cwd = str(sys.argv[0])
        cwd, pyname = os.path.split(path)
    else:
        cwd = path
    os.chdir(cwd)
    return cwd





