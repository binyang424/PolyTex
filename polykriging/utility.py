import sys, os

######################################################
#                               Current work directory                                   #
######################################################
def cwd_chdir(path = ""):
    '''
    Set given directory or the folder where the code file is as working directory

    Returns
    -------
    cwd : path of current working directory 
    '''
    import sys, os
    if path == "":
        cwd = str(sys.argv[0])
        cwd, pyname = os.path.split(path)
    else:
        cwd = path
    os.chdir(cwd)
    return cwd

######################################################
#                                File list in the given folder                             #
######################################################
def filenames(path, condition = "csv"):
    # Creat a filenames list for all test data.
    filenamels = os.listdir(path)
    flst=[];   # file list
    flst = [x for x in filenamels if (condition in x)]
    flst.sort();
    return flst

######################################################
#                            Create csv file with given info.                         #
######################################################
def create_csv(path, filename, coordinate, csv_head = ["line","x", "y", "z"]):
    """
    coordinate: [[x,y,z]]
    
    """
    path = path + filename + ".csv"
    # 添加newline参数避免输出换行符
    with open(path,'w', newline="") as f:  
        csv_write = csv.writer(f)
        csv_head = ["line","x", "y", "z", "distance", "normalised distance", "angle position", "radius", "nugget"]
        csv_write.writerow(csv_head)
        for element in coordinate:
            csv_write.writerow(element)
    return 1


######################################################
#          Choose a directory with GUI  and return its path              #
######################################################
from tkinter import Tk, filedialog
import time, os

def choose_directory(title ='Select the directory of selections in ".csv" format:'):
    print(title)
    # pointing root to Tk() to use it as Tk() in program.
    # like a window (container) where we can put widgets.
    directory_root = Tk()                    
    directory_root.withdraw()           # Hides small tkinter window.
    directory_root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
    path_work = filedialog.askdirectory() # Returns opened path as str
    if path_work == '':
        print('You did not select any folder! Please select again.')
        time.sleep(2)
        return choose_directory()
    else:
        # askdirectory 获得是 正斜杠 路径C:/，所以下面要把 / 换成 反斜杠\
        return path_work.replace('/', os.sep)

######################################################
#                       Export and import expression                               #
######################################################
def expr_io(filename, expression=''):
    '''
    Import or export kriging expression as .txt file.

    Parameters
    ----------
    filename : String
        filename with extension txt.
    expression : expression, optional
        Import txt file which contains a kriging expression if Empty string '' is taken. 
        Export the kriging expression when expression is given as a variable name of kriging expression.
        The default is ''.
    '''
    import sympy as sym
    
    if expression!="":
        with open(filename,"w") as f:
            f.write(str(expression))
            f.close
            print('The kriged function has been saved.')
    else:
        file = open(filename,"r")
        expression = file.readlines()
        return expression[0]
