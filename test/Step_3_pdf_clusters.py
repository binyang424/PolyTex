"""
PDF-based point clustering
==================================

This example shows how to estimate the probability density function (pdf) of the
input data and cluster the data points based on the pdf analysis.

Kernel density estimation (KDE) is a non-parametric way to estimate the probability
density function of a random variable. The extrema of the pdf are used as the cluster
centers. The initial bandwidth of the kernel is estimated by Scott's rule.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import polytex as ptx
import os

# Conversion factor from pixel to mm
resolution = 0.022  # 0.022 mm
# number of control points for contour description
extremaNum, windows, nuggets = 30, 5, [1e-3]

######################################################################
# Data loading
# ---------------------
path = ptx.choose_file(titl="Directory for the file containing "
                           "sorted coordinates (.coo)", format=".coo")
filename = os.path.basename(path)
coordinatesSorted = ptx.pk_load(path)

######################################################################
# Initial bandwidth estimation by Scott's rule
# --------------------------------------------
# Using Scott's rule, an initial bandwidth for kernel density estimation
# is calculated from the standard deviation of the normalized distances
# in coordinatesSorted.
t_norm = coordinatesSorted["normalized distance"]
std = np.std(t_norm)
bw = ptx.stats.bw_scott(std, t_norm.size) / 2
print("Initial bandwidth: {}".format(bw))

######################################################################
# Kernel density estimation
# --------------------------------------------
# kdeScreen method from ptx.stats is used to find the kernel density
# estimation for a linear space spanning from 0 to 1 with 1000 points.
t_test = np.linspace(0, 1, 1000)
clusters = ptx.stats.kdeScreen(t_norm, t_test, bw, plot=False)

# log-likelihood
LL = ptx.stats.log_likelihood(clusters["pdf input"])

######################################################################
# Save pdf analysis
# --------------------------------------------
# Results from the KDE are saved with a filename that includes cluster
# centers information and the computed bandwidth.
cluster_centers = clusters["cluster centers"]
ptx.pk_save(filename[:-4] + "_clusters" + str(len(cluster_centers)) +
           "_bw" + str(round(bw, 3)) + ".stat", clusters)

######################################################################
# Reload pdf analysis results
# --------------------------------------------
# The previously saved statistical data can be reloaded as follows:
reload = ptx.pk_load(filename[:-4] + "_clusters" + str(len(cluster_centers)) +
                    "_bw" + str(round(bw, 3)) + ".stat")

######################################################################
# Plot pdf analysis
# --------------------------------------------
# The pdf analysis is plotted with the scatters colored by the radial
# normalized distance (ax1) and the cluster labels (ax2).
plt.close('all')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
# fig.subplots_adjust(bottom=0.5)
cmap1 = mpl.cm.cool
cmap2 = mpl.cm.jet

""" color the scatters with the radial normalized distance """
ax1.scatter(coordinatesSorted["X"], coordinatesSorted["Y"], s=25,
            c=clusters["t input"], cmap=cmap1, alpha=1 / 2, edgecolors='none')

""" color the scatters with cluster labels """
# colorize the scatter plot according to clusters
color = ptx.color_cluster(clusters)
ax2.scatter(coordinatesSorted["X"], coordinatesSorted["Y"], s=25,
            c=color, cmap=cmap2, alpha=1 / 2, edgecolors='none', label="bw = %.2f" % bw)

ax2.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_ylabel('y (mm)')
# remove the ticks in ax1
ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
ax1.set_aspect(2)  # aspect ratio: y/x
ax2.set_aspect(2)  # aspect ratio: y/x
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

""" colorbar """
fig2, ax1 = plt.subplots(figsize=(6, 1))
fig2.subplots_adjust(bottom=0.5)

bounds = np.arange(0, len(cluster_centers))
norm = mpl.colors.BoundaryNorm(bounds, cmap2.N,
                               # extend='both'
                               )

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap2),
             cax=ax1, orientation='horizontal', label='pdf')
plt.show()
