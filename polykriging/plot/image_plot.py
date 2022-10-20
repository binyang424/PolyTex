# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt


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
