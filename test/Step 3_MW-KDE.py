"""
Step 3: Moving window KDE
=========================

This example shows how to use the moving window KDE to resample the control
points of the tow.

"""
###############################################################################
# Load example dataset
# -----------------------------------------------------------------------------
import numpy as np
import polykriging as pk
import matplotlib.pyplot as plt

# Input: parameters
resolution = 0.022  # 0.022 mm
# number of extrema (control points) for contour description
extremaNum, windows, nuggets = 10, 2, [1e-3]

path = pk.example("sorted coordinate")
coordinatesSorted = pk.pk_load(path)

###############################################################################
# Visualize the dataset (a tow contour)
# -----------------------------------------------------------------------------
# The tow contour is described by a set of control points. The control points
# can be labeled by its z coordinate (the scaning slices) since the dataset is
# obtained from Micro CT scanning. The control points are sorted by its z coordinate.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = coordinatesSorted["X"].to_numpy()
y = coordinatesSorted["Y"].to_numpy()
z = coordinatesSorted["Z"].to_numpy()
ax.scatter(x, y, z, s=1, marker="o", c=z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Control points of a fiber tow")
ax.set_aspect("equal")
plt.show()

###############################################################################
# Slice number of the tow
# -----------------------------------------------------------------------------
slices = np.array(coordinatesSorted["Z"] / resolution, dtype=int)
nslices = np.unique(slices).size  # number of slices


###############################################################################
# Dataset preparation for moving window kernel density estimation
# -----------------------------------------------------------------------------
t_norm = np.vstack((coordinatesSorted["normalized distance"], slices)).T

# bw = np.arange(0.01, 0.03, 0.01)  # specify a range for bandwidth optimization
# initialize the bandwidth according to Scott's rule
bw = 0.01

kdeOutput, cluster_centers = pk.stats.movingKDE(t_norm, bw, windows, extremaNum)

kdeOutput.plot(x="normalized distance", y="probability density")
plt.show()
