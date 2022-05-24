from polykriging import utility
import numpy as np
import zipfile, os

# 压缩某个目录下所有文件
def compress_file(zipfilename, dirname):
    # zipfilename是压缩包名字，dirname是要打包的目录
    if os.path.isfile(dirname):
        with zipfile.ZipFile(zipfilename, 'w') as z:
            z.write(dirname)
    else:
        with zipfile.ZipFile(zipfilename, 'w') as z:
            for root, dirs, files in os.walk(dirname):
                for single_file in files:
                    if single_file != zipfilename:
                        filepath = os.path.join(root, single_file)
                        z.write(filepath)

path = utility.choose_directory(titl =
                                "Choose the directory that contains ROIs stored in csv files")
cwd = utility.cwd_chdir(path)
files = utility.filenames(path)

yarn = files[0][:files[0].rfind("_")]

compress_file(yarn + ".zip", "./")
for i in np.arange(len(files)):
    # File loading order control
    section = files[0][ : files[0].rfind("_") + 1] + str(i+1) + ".csv"
    csPoints = np.loadtxt(section, comments= section,
                            delimiter=",", skiprows=1
                            )
    try:
        surfPoints = np.vstack((surfPoints, csPoints))
    except NameError:
        surfPoints = csPoints
    os.remove(section)  # 处理后删除文件

# save the csv files into a npy file
# [original point order, X, Y, Z (SLICE NUMBER)]
np.save(yarn, surfPoints)
