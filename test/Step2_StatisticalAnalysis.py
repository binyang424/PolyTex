import numpy as np
import scipy
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

import sympy as sym
from polykriging import curve2D, utility


#path = r"C:\BinY\DMT\Code\processedData\Vf60\warp\yarn_1.npz"
path = r"C:\Users\palme\Desktop\Code\processedData\Vf60\warp\yarn_1.npz"
pcdRaw = np.load(path)
fileList = list(pcdRaw.keys())
yarn = 1

for i in range(len(fileList) - 1):
    slicedata = "yarn_" + str(yarn) + "_coordinateSorted_" + str(i) + ".npy"
    try:
        pcd = np.vstack((pcd,pcdRaw[slicedata]))
    except NameError:
        pcd = pcdRaw[slicedata]
    if i == 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        # ax.set_ylabel('Normalized distance')
    # The following angle positions should be in radians.
    ax.scatter(pcdRaw[slicedata][:, 2]/360*2*np.pi, pcdRaw[slicedata][:,1],
               alpha = 0.7, s = 1.5 )

# reference line for a circle:
ax.plot(np.arange(0, 2*np.pi, 2*np.pi/360), np.arange(0,1,1/360), linestyle='--', color = 'red' )      
# reference line for a square:

#plt.show()

# geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
#       centroidX, centroidY, centroidZ]
pcd = pcd    
# [distance, normalized distance, angular position (degree),
#       coordinateSorted(X, Y, Z)]
geomFeatures = pcdRaw["yarn_1_geomFeatures.npy"]






















# for yarnIndex in np.arange(len(tows)):
for yarnIndex in np.arange(0, 1):
    key = 'geomParametric_' + str(yarnIndex)
    # ["iSlice", "x", "y", "z", "distance", "normalised distance", "angle position (degree)"]
    geomParametric = geomParametric_weft[key]
    distance_norm = geomParametric[:, 5]  # normalised distance
    angle = geomParametric[:, 6]  # angle

    #bw = np.arange(0.1, 0.00001, -0.00005)
    bw = np.array([0.089, 0.053, 0.072, 0.079, 0.099,0.1, 0.066, 0.064])
    extremaNum = 30
    nslice = np.max(geomParametric[:, 0])
    windowWidth = 23
    # for windowIndex in np.arange(1,2):
    for windowIndex in np.arange(int(nslice / windowWidth) + 1):
        windowMedian = windowWidth * windowIndex + windowWidth / 2
        kde_array = distance_norm[abs(geomParametric[:, 0] - windowMedian) <= windowWidth / 2.]

        print("kde_array", kde_array[:4])

        xWindow, yWindow = movingKDE(kde_array, [bw[windowIndex]], windowIndex, extremaNum)
        print(windowIndex, windowMedian, xWindow[0])

        xWindow = np.hstack((np.array([0]), xWindow))

        xLocal = np.zeros([xWindow.shape[0], 3])
        xLocal[:, 0] = windowIndex
        xLocal[:, 1] = windowMedian  # windowWidth = xLocal[0,1] *2
        xLocal[:, 2] = xWindow

        try:
            x = np.vstack((x, xLocal))
        except NameError:
            x = xLocal

    x[:, 2][x[:, 2] < 0] = abs(x[:, 2][x[:, 2] < 0])
    np.save('./FabricSm/weft/kde_distance_norm_' + str(yarnIndex), x)

# import json
# with open('xkde.json', 'w', encoding='utf-8') as f:
#     json.dump(xcompr, f, ensure_ascii=True, indent=2)
# f.close()


#
# #Show the plot
# plt.draw()
#
# # # Save to a File
# # filename = 'myplot'
# # plt.savefig(filename + '.pdf',format = 'pdf', transparent=True)
