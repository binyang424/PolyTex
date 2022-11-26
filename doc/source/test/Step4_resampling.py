"""
Resampling
=================

Test

"""

import matplotlib.pyplot as plt
from polykriging.kriging import curve2D
import numpy as np
import polykriging as pk

"""   Overfitting test    """
ii = 0
for iSlice in [250 * 0.022]:
    # for iSlice in slices[0:6]:
    print(ii)
    ii += 1
    # return the index of the slice satisfying the condition
    index = np.where(coordinatesSorted[:, -1] == iSlice)
    # average of the index
    indexAvg = int(np.average(index))

    # extrema: 行为窗口，0列为包含在该窗口的点的终止序号，其余列为极值点
    for i in range(cluster_centers[:, 0].size):  # 即遍历每一个窗口
        if indexAvg < cluster_centers[i, 0]:
            interp = kdeOutput[kdeOutput[:, 0] == i, 1][np.int32(cluster_centers[i, 1:])]

    mask = coordinatesSorted[:, -1] == iSlice
    # 选出当前slice的归一化距离和坐标xyz数据
    coordinate = coordinatesSorted[:, [1, -3, -2, -1]][mask]
    # 对数据进行线性插值，一遍提高Kriging的精度
    coordinate = curve2D.addPoints(coordinate, threshold=0.02)

    plt.close('all')
    ax1 = plt.subplot(1, 1, 1)

    for nugget in nuggets:
        # 插值结果，插值表达式
        # xinter, xexpr = curve2D.curve2Dinter(coordinate[:, [0, 1] ],
        #                              'lin', "lin", nugget, interp )
        # yinter, yexpr = curve2D.curve2Dinter(coordinate[:, [0, 2] ],
        #                              'lin', "lin", nugget, interp )

        # Split the data to improve interpolation quality
        # and the efficiency of kriging
        mask1 = coordinate[:, 0] < 0.5
        mask2 = coordinate[:, 0] >= 0.5
        xinter, xexpr = curve2D.curve2Dinter(coordinate[:, [0, 1]][mask1],
                                             'lin', "lin", nugget, interp[interp < 0.5])
        yinter, yexpr = curve2D.curve2Dinter(coordinate[:, [0, 2]][mask1],
                                             'lin', "lin", nugget, interp[interp < 0.5])
        xinterSplit, _ = curve2D.curve2Dinter(coordinate[:, [0, 1]][mask2],
                                              'lin', "lin", nugget, interp[interp >= 0.5])
        yinterSplit, _ = curve2D.curve2Dinter(coordinate[:, [0, 2]][mask2],
                                              'lin', "lin", nugget, interp[interp >= 0.5])
        xinter = np.hstack((xinter, xinterSplit))
        yinter = np.hstack((yinter, yinterSplit))

        # ax1.plot(xinter, yinter, '--', label = str(nugget), linewidth = 1)
        ax1.scatter(xinter, yinter, s=40, marker='+', color='red')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.invert_yaxis()  # y轴反向

    ax1.fill(coordinate[:, 1], coordinate[:, 2], alpha=0.3, color='pink')

    plt.legend()
    ax1.axis("equal")
    plt.show()

''' Polar plot: angular position - normalized distance '''
# #fig = plt.figure()
# #ax = fig.add_subplot(projection='polar')
# ## ax.set_ylabel('Normalized distance')
# ## The following angle positions should be in radians.
# #ax.scatter(coordinatesSorted[:, 2]/360*2*np.pi, coordinatesSorted[:,1],
# #           alpha = 0.7, s = 1 )
# ## reference line for a circle:
# #ax.plot(np.arange(0, 2*np.pi, 2*np.pi/360), np.arange(0,1,1/360), linestyle='--', color = 'red' )
# ##plt.show()
