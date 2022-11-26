"""
Moving window KDE
=================

Test

"""


import numpy as np
import polykriging as pk
import matplotlib.pyplot as plt

# Input: parameters
resolution = 0.022  # 0.022 mm
# number of extrema (control points) for contour description
extremaNum, windows, nuggets = 10, 1, [1e-3]

''' Data loading '''
path = pk.fileio.choose_file(titl="Directory for file CoordinatesSorted file (.coo)", format=".coo")
coordinatesSorted = pk.pk_load(path)

# read Y column from coordinatesSorted and convert to numpy array in int
slices = np.array(coordinatesSorted["Y"] / resolution, dtype=int)

nslices = np.unique(slices).size
t_norm = np.vstack((coordinatesSorted["normalized distance"], slices)).T

# bw = np.arange(0.01, 0.03, 0.01)  # specify a range for bandwidth optimization
# initialize the bandwidth according to Scott's rule
bw = 0.01

kdeOutput, cluster_centers = pk.stats.movingKDE(t_norm, bw, windows, extremaNum)

kdeOutput.plot(x="normalized distance", y="probability density")
plt.show()