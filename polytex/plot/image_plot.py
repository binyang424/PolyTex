# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.5, alpha=1):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    url : 
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Parameters
    ----------
    color : str or tuple
        color to lighten
    amount : float
        amount to lighten the color. Value less than 1 produces a lighter color,
        value greater than 1 produces a darker color.
    alpha : float
        alpha value of the color. Default is 1. The alpha value is a float
        between 0 and 1.

    Returns
    -------
    tuple
        modified color in RGBA tuple (float values in the range 0-1).
        
    Examples:
    >> lighten_color('g', 0.3, 1)
    (0.5500000000000002, 0.9999999999999999, 0.5500000000000002, 1)
    >> lighten_color('#F034A3', 0.6, 0.5)
    (0.9647058823529411, 0.5223529411764707, 0.783529411764706, 0.5)
    >> lighten_color((.3,.55,.1), 0.5)
    (0.6365384615384615, 0.8961538461538462, 0.42884615384615377, 1)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    color_lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    
    return (*color_lightened, alpha)
    
    
def para_plot():
    """
    This function is used to describe the parameters of the plot.
    """
    import matplotlib as mpl
    plt.style.use(['science', 'high-vis', 'grid'])
    params = {
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
    }

    font = {'family': 'normal', 'weight': 'normal'}
    plt.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"

    mpl.rcParams.update(params)


def vert_sub_plot(num_plots, vspace, x, y, labels):
    """
    This function is used to plot multiple subplots vertically.

    Parameters
    ----------
    num_plots : int
        The number of subplots.
    vspace : float
        The vertical space between subplots.
    x : numpy array
        The x-axis data. The shape of x should be (num_points, num_plots).
    y : numpy array
        The y-axis data. The shape of y should be (num_points, num_plots).
    labels : list
        The labels of subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    para_plot()
    fig, axs = plt.subplots(1, num_plots, sharey=True)

    # Remove horizontal space between axes
    fig.subplots_adjust(wspace=vspace)

    for i in range(num_plots):
        axs[i].plot(x[:, i], y[:, i], label=labels[i], linewidth=2)
        # axs[i].set_yticks(np.arange(0, 1.0, 0.2))
        # axs[i].set_xticks(np.arange(0, 2.0, 0.5))
        # axs[i].set_ylim(0, 1)
        axs[i].tick_params(axis='both', which='both', length=0)

    axs[0].set_ylabel("Normalized distance")
    # axs[1].set_xlabel("Density")

    # plt.title('Density', x=-0.65, y=- 0.15, fontsize=12)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1)

    return fig


def plot_on_img(x, y, backgroundImg, labels=[], save=False):
    """
    This function is used to plot the image.

    Parameters
    ----------
    x,y: numpy array
        if x.shape[1]>1, 一个背景多个图
    labels: list of string
        , 多个图的legend
    img:
        a image as the background
    save: bool
        if True, save the image, default is False.

    Returns
    -------
    None.
    """
    img = cv2.imread(backgroundImg)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_bin = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # cv2.threshold (源图片, 阈值, 填充色, 阈值类型)
    fig = plt.figure(figsize=(16 / 2.54, 9.5 / 2.54))
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.83])
    ax.imshow(imgray, cmap='gray', vmin=0, vmax=255)
    ##ax.imshow(img_bin[1], origin='lower',  cmap='gray', vmin=0, vmax=255)

    # TODO 一个背景图多个曲线图
    x_shape, y_shape = len(np.shape(x)), len(np.shape(y))
    if x_shape == 2 or y_shape == 2:
        pass
    elif x_col == 1 and y_col == 1:
        # Append values to the end: np.append(arr, values, axis=None)
        x = np.append(x, [x[0]], axis=0)
        y = np.append(y, [y[0]], axis=0)
        ax.plot(x, y, '-.', label=labels, color='r')
    else:
        print('Only 1D or 2D (for multi curve plots) numpy array is accepted for x and y.')
    plt.show()


def xy_interp(*axis_list, num=100, raw=False):
    """
    Interpolate the axis_list to the same x-axis and calculate the mean y-axis
    for all the input x-y pairs (midline).

    TODO: check what happens if the range of x-axis is not the same for all the input axis_list.

    Note:
    -----
    [algorithm - How to interpolate a line between two other lines in python]
    (https://stackoverflow.com/questions/49037902/how-to-interpolate-a-line-between-two-other-lines-in-python/49041142#49041142)

    Parameters
    ----------
    axis_list : list of np.ndarray
        Each element is a 2D array with shape (n, 2), where n is the number of points.
        The first column is x-axis and the second column is y-axis.
    num : int, optional
        The number of points to interpolate. The default is 100.
    raw : bool, optional
        If True, return the raw interpolated axis_list. The default is False.

    Returns
    -------
    mid : np.ndarray
        The interpolated midline with shape (n, 2), where n is the number of points.
        The first column is x-axis and the second column is y-axis.
    raw_interp : np.ndarray
        The raw interpolated axis_list with shape (n, m), where n is the number of points
        and m is the number of input axis_list. The first m columns are x-axis and the
        last m columns are y-axis.

    Examples
    --------
    >>> x1 = np.linspace(0, 10, 10)
    >>> y1 = np.linspace(0, 15, 10)
    >>> x2 = np.linspace(0, 10, 20)
    >>> y2 = np.linspace(0, 12, 20)
    >>> x3 = np.linspace(0, 9, 30)
    >>> y3 = np.linspace(0, 18, 30)
    >>> interp(np.vstack((x1, y1)).T, np.vstack((x2, y2)).T, np.vstack((x3, y3)).T)
    """
    min_max_xs = [(min(axis[:, 0]), max(axis[:, 0])) for axis in axis_list]

    # TODO : The interpolated range of x-axis is not the same for all the input axis_list.
    # 1. Interpolate the axis_list to the same x-axis: [maximun min_x, minimun max_x].
    # 2. Calculate the mean y-axis for all the input x-y pairs (midline)
    new_axis_xs = [np.linspace(min_x, max_x, num) for min_x, max_x in min_max_xs]
    new_axis_ys = [np.interp(new_x_axis, axis[:, 0], axis[:, 1])
                   for axis, new_x_axis in zip(axis_list, new_axis_xs)]

    midx = [np.mean([new_axis_xs[axis_idx][i]
                     for axis_idx in range(len(axis_list))]) for i in range(num)]
    midy = [np.mean([new_axis_ys[axis_idx][i]
                     for axis_idx in range(len(axis_list))]) for i in range(num)]

    mid = np.vstack((midx, midy)).T
    print(np.array(new_axis_xs).shape, np.array(new_axis_ys).shape)
    raw_interp = np.vstack((new_axis_xs[0], new_axis_ys))

    if raw:
        return mid, raw_interp
    else:
        return mid
        

if __name__ == '__main__':
    img = './test/imagePlot/trans0000.tif'
    data = './test/imagePlot/4_1_XY_Coordinates.csv'
    coordinate = np.loadtxt(open(data, "rb"), delimiter=",",
                            skiprows=1, usecols=(0, 1))
    imagePlot(coordinate[:, 0], coordinate[:, 1], img)

    # vSubPlots()
    t = np.arange(0.0, 2.0, 0.01).reshape(-1, 1)

    s1 = np.sin(2 * np.pi * t).reshape(-1, 1)
    s2 = np.exp(-t).reshape(-1, 1)
    s3 = s1 * s2.reshape(-1, 1)

    s = np.hstack((s1, s2, s3))
    t = np.hstack((t, t, t))

    labels = ["Global KDE", 'MW-KDE (5)', 'MW-KDE (10)']

    fig = vSubPlots(3, 0, t, s, labels)

    fig.savefig("image.jpg", dpi=600)
