def zipFilelist(path, fileList, filename, removeFile="True"):
    """
    Add multiple files to the zip file.
    :param path: the path of the files to be added to zip file
    :param fileList: the list of files to be added to the zip file
    :param filename: the name of the zip file
    :param removeFile: whether to remove original files after adding to zip file
    :return: None.
    """
    from zipfile import ZipFile
    import os
    with ZipFile(filename + '.zip', 'w') as zipObj:
        for i in range(len(fileList)):
            zipObj.write(path + fileList[i])

            if removeFile == "True":
                os.remove(path + fileList[i])


def cwd_chdir(path=""):
    """
    Set given directory or the folder where the code file is as current working directory
    :param path: the path of current working directory. if empty, the path of the code file is used.
    :return cwd: the current working directory.
    """
    import sys, os
    if path == "":
        cwd = str(sys.argv[0])
        cwd, pyname = os.path.split(path)
    else:
        cwd = path
    os.chdir(cwd)
    return cwd


def filenames(path, condition="csv"):
    """
    Get the list of files in the given folder.
    :param path: the path of the folder
    :param condition: filter for file selection.
    :return flst: the list of files in the given folder.
    """
    import os
    filenamels = os.listdir(path)
    # filter the file list by the given condition.
    flst = [x for x in filenamels if (condition in x)]
    flst.sort()
    return flst


######################################################
#                            Create csv file with given info.                         #
######################################################
def create_csv(path, filename, coordinate, csv_head=["line", "x", "y", "z"]):
    """
    coordinate: [[x,y,z]]
    
    """
    path = path + filename + ".csv"
    # 添加newline参数避免输出换行符
    with open(path, 'w', newline="") as f:
        csv_write = csv.writer(f)
        # csv_head = ["line", "x", "y", "z", "distance",
        #       "normalised distance", "angle position", "radius", "nugget"]
        csv_write.writerow(csv_head)
        for element in coordinate:
            csv_write.writerow(element)
    return 1


def choose_directory(titl='Select the directory of selections in ".csv" format:'):
    """
    Choose a directory with GUI and return its path.
    :param titl: String. The title of the window.
    :return path: String. The path of the directory.
    """
    from tkinter import Tk, filedialog
    import os,time

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


def choose_file(titl='Select the directory of selections in ".csv" format:'):
    """
    Choose a file with GUI and return its path.
    :param titl: String. The title of the window.
    :return path: String. The path of the file.
    """
    from tkinter import Tk, filedialog
    import os,time
    
    print(titl)
    directory_root = Tk()
    directory_root.withdraw()  # Hides small tkinter window.
    directory_root.attributes('-topmost',
                        True)  # Opened windows will be active. above all windows.
    path_work = filedialog.askopenfilename(title=titl)  # Returns opened path as str

    # replace the forward slash returned by askdirectory
    # with backslash (\) on Windows.
    return path_work.replace('/', os.sep)


def expr_io(filename, expression=''):
    """
    Import or export kriging expression as .txt file.
    :param filename: String. filename with extension txt.
    :param expression: String. kriging expression.
        Import txt file which contains a kriging expression if Empty string '' is taken.
        Export the kriging expression when expression is given as a variable name of
        kriging expression. The default is ''.
    :return: String. kriging expression.
    """
    if expression != "":
        with open(filename, "w") as f:
            f.write(str(expression))
            f.close
            print('The kriged function has been saved.')
    else:
        file = open(filename, "r")
        expression = file.readlines()
        return expression[0]