'''
https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html
'''

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from polykriging import utility

plt.style.use(['science','ieee'])
#%% Data loading
path = utility.choose_file(titl = 
                           "Directory for .npz file containing GeometryFeatures and CoordinatesSorted")
pcdRaw = np.load(path)

# geomFeature = [Area, Perimeter, Width, Height, AngleRotated, Circularity,
#       centroidX, centroidY, centroidZ]
# coordinateSorted = [distance, normalized distance, angular position (degree),
#       coordinateSorted(X, Y, Z)]
coordinatesSorted = pcdRaw["coordinatesSorted.npy"]
geomFeatures = pcdRaw["geomFeatures.npy"]

obs_dist = coordinatesSorted[:,1]
#%% Histogram - The simplest non-parametric technique for density estimation
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

# Plot the samples
s = int(obs_dist.size/150)
scatterS = np.random.choice(obs_dist, s)
ax.scatter(
    scatterS,
    np.abs(np.random.randn(s)) / 10,
    marker="x",
    color="red",
    zorder=20,
    label="Control points",
    alpha=0.5,
)

# Plot the histrogram
ax.hist(
    obs_dist,
    bins=80,
    density=True,
    label="Histogram from samples",
    zorder=5,
    edgecolor="k",
    alpha=0.4,
)

#%%
# Plot the KDE as fitted using the default arguments
kde = sm.nonparametric.KDEUnivariate(obs_dist)
kde.fit()  # Estimate the densities
# Plot the KDE for various bandwidths
for bandwidth in [0.005, 0.01, 0.1]:
    kde.fit(bw=bandwidth)  # Estimate the densities
    ax.plot(
        kde.support,
        kde.density,
        "--",
        lw=1.5,
        #color="k",
        zorder=10,
        label="KDE from samples, bw = {}".format(round(bandwidth, 3)),
    )

ax.set_xlim([0,1])
ax.set_xlabel("Normalized distance")
ax.legend(loc="best")
ax.grid(True, zorder=-5)

#%% Comparing kernel functions
# from statsmodels.nonparametric.kde import kernel_switch

# list(kernel_switch.keys())

# # Create a figure
# fig = plt.figure(figsize=(12, 5))

# # Enumerate every option for the kernel
# for i, (ker_name, ker_class) in enumerate(kernel_switch.items()):

#     # Initialize the kernel object
#     kernel = ker_class()

#     # Sample from the domain
#     domain = kernel.domain or [-3, 3]
#     x_vals = np.linspace(*domain, num=2 ** 10)
#     y_vals = kernel(x_vals)

#     # Create a subplot, set the title
#     ax = fig.add_subplot(3, 3, i + 1)
#     ax.set_title('Kernel function "{}"'.format(ker_name))
#     ax.plot(x_vals, y_vals, lw=3, label="{}".format(ker_name))
#     ax.scatter([0], [0], marker="x", color="red")
#     plt.grid(True, zorder=-5)
#     ax.set_xlim(domain)

# plt.tight_layout()
