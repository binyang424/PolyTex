import csv
import numpy as np

def save_csv(path, filename, coordinate, csv_head=["line", "x", "y", "z"]):
    """
    Save numpy array to csv file with given info in the first row.

    Parameters
    ----------
    path:
        the path of the folder
    filename:
        the name of the csv file
    coordinate:
        the list of coordinates to be saved in the csv file
    csv_head:
        the list of the head of the csv file

    Returns
    -------
    None.

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