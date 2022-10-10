# ！/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import Tk, filedialog
import os, time


def choose_directory(titl='Select the directory of selections in ".csv" format:'):
    """
    Choose a directory with GUI and return its path.

    Parameters
    ----------
    titl: String.
        The title of the open folder dialog window.

    Returns
    -------
    path: String. The path of the directory.
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
        print('You did not select any folder! Please select again.')
        time.sleep(2)
        return choose_directory()
    else:
        # replace the forward slash returned by askdirectory
        # with backslash (\) on Windows.
        return path_work.replace('/', os.sep)


def filenames(path, condition="csv"):
    """
    Get the list of files in the given folder.

    Parameters
    ----------
    path:
        the path of the folder
    condition:
        filter for file selection.

    Returns
    -------
    flst: the list of files in the given folder.
    """
    filenamels = os.listdir(path)
    # filter the file list by the given condition.
    flst = [x for x in filenamels if (condition in x)]
    flst.sort()
    return flst


def zipFilelist(path, fileList, filename, removeFile="True"):
    """
    Add multiple files to the zip file.

    Parameters
    ----------
    path:
        the path of the files to be added to zip file
    fileList :
        the list of files to be added to the zip file
    filename:
        the name of the zip file
    removeFile:
        whether to remove original files after adding to zip file

    Returns
    -------
    :return: None.
    """
    from zipfile import ZipFile
    import os
    with ZipFile(filename + '.zip', 'w') as zipObj:
        for i in range(len(fileList)):
            zipObj.write(path + fileList[i])

            if removeFile == "True":
                os.remove(path + fileList[i])


def choose_file(titl='Select the directory of selections in ".csv" format:'):
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
                              True)  # Opened windows will be active. above all windows.
    path_work = filedialog.askopenfilename(title=titl)  # Returns opened path as str

    # replace the forward slash returned by askdirectory
    # with backslash (\) on Windows.
    return path_work.replace('/', os.sep)


