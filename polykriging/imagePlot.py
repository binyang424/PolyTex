import numpy as np
import cv2
from matplotlib import pyplot as plt

def imagePlot(x, y, backgroundImg, labels = [], save = False):
    '''
    x,y numpy array, if x.shape[1]>1, 一个背景多个图
    label: list of string, 多个图的legend
    img: a image as the background
    '''
    img = cv2.imread(backgroundImg)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_bin = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # cv2.threshold (源图片, 阈值, 填充色, 阈值类型)
    fig=plt.figure(figsize=(16/2.54,9.5/2.54))
    ax = fig.add_axes([0.12,0.1,0.85,0.83])
    ax.imshow(imgray,  cmap='gray', vmin=0, vmax=255)
    ##ax.imshow(img_bin[1], origin='lower',  cmap='gray', vmin=0, vmax=255)
    
    #TODO 一个背景图多个曲线图
    x_shape, y_shape = len(np.shape(x)), len(np.shape(y))
    if x_shape==2 or y_shape==2:
        pass
    elif x_col==1 and y_col==1:
        # Append values to the end: np.append(arr, values, axis=None)
        x = np.append(x, [x[0]], axis = 0)
        y = np.append(y, [y[0]], axis = 0)
        ax.plot(x, y, '-.', label = labels, color = 'r')
    else:
        print('Only 1D or 2D (for multi curve plots) numpy array is accepted for x and y.')
    plt.show()


if __name__ == '__main__':
    img = './test/imagePlot/trans0000.tif'
    data = './test/imagePlot/4_1_XY_Coordinates.csv'
    coordinate = np.loadtxt(open(data, "rb"), delimiter=",",
                            skiprows=1, usecols=(0,1))
    imagePlot(coordinate[:,0], coordinate[:,1], img)
