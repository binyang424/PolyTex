import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from polykriging import utility
import polykriging as pk

# Input: parameters
resolution = 0.022  # 0.022 mm
# number of extrema (control points) for contour description
extremaNum, windows, nuggets = 30, 5, [1e-3]

''' Data loading '''
path = utility.choose_file(titl="Directory for file CoordinatesSorted file (.coo)")
coordinatesSorted = pk.pk_load(path).to_numpy()

''' Initial bandwidth estimation by Scott's rule '''
std = np.std(coordinatesSorted[:, 1])
bw = pk.stats.bw_scott(std, coordinatesSorted[:, 1].size)

'''  Kernel density estimation   '''
t_test = np.linspace(0, 1, 1000)
clusters = pk.stats.kdeScreen(coordinatesSorted[:, 1], t_test, 0.05, plot=True)

# log-likelihood
LL = pk.stats.log_likelihood(clusters["pdf input"])

# colorize the scatter plot according to clusters
color = pk.color_cluster(clusters)

# plot scatter plot
plt.close('all')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
# fig.subplots_adjust(bottom=0.5)
cmap1 = mpl.cm.cool
cmap2 = mpl.cm.jet

# color the scatters with the density pdf
ax1.scatter(coordinatesSorted[:, 3], coordinatesSorted[:, 4], s=30,
            c=clusters["t input"], cmap=cmap1, alpha=1/10, edgecolors='none')

# color the scatters with the density pdf
ax2.scatter(coordinatesSorted[:, 3], coordinatesSorted[:, 4], s=30,
            c=color, cmap=cmap2, alpha=1/20, edgecolors='none')

ax2.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_ylabel('y (mm)')
# remove the ticks in ax1
ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
ax1.set_aspect(2.5)  # aspect ratio: y/x
ax2.set_aspect(2.5)  # aspect ratio: y/x
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

""" colorbar """
cluster_centers = clusters["cluster centers"]
plt.close('all')
fig, ax1 = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

bounds = np.arange(0, len(cluster_centers) + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap2.N,
                               # extend='both'
                               )

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax1, orientation='horizontal', label='pdf')
plt.show()