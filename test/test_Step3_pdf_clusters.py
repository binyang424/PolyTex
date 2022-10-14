import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import polykriging as pk
import os

# Input: parameters
resolution = 0.022  # 0.022 mm
# number of extrema (control points) for contour description
extremaNum, windows, nuggets = 30, 5, [1e-3]

''' Data loading '''
path = pk.choose_file(titl="Directory for the file containing "
                           "sorted coordinates (.coo)", format=".coo")
filename = os.path.basename(path)
coordinatesSorted = pk.pk_load(path)

''' Initial bandwidth estimation by Scott's rule '''
t_norm = coordinatesSorted["normalized distance"]
std = np.std(t_norm)
bw = pk.stats.bw_scott(std, t_norm.size) / 3
print("Initial bandwidth: {}".format(bw))

'''  Kernel density estimation   '''
t_test = np.linspace(0, 1, 1000)
clusters = pk.stats.kdeScreen(t_norm, t_test, bw, plot=True)

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

""" color the scatters with the radial normalized distance """
ax1.scatter(coordinatesSorted["X"], coordinatesSorted["Y"], s=25,
            c=clusters["t input"], cmap=cmap1, alpha=1 / 2, edgecolors='none')

""" color the scatters with cluster labels """
ax2.scatter(coordinatesSorted["X"], coordinatesSorted["Y"], s=25,
            c=color, cmap=cmap2, alpha=1 / 2, edgecolors='none', label="bw = %.2f" % bw)

ax2.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax2.set_ylabel('y (mm)')
# remove the ticks in ax1
ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
ax1.set_aspect(1)  # aspect ratio: y/x
ax2.set_aspect(1)  # aspect ratio: y/x
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

""" colorbar """
cluster_centers = clusters["cluster centers"]
fig2, ax1 = plt.subplots(figsize=(6, 1))
fig2.subplots_adjust(bottom=0.5)

bounds = np.arange(0, len(cluster_centers))
norm = mpl.colors.BoundaryNorm(bounds, cmap2.N,
                               # extend='both'
                               )

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap2),
             cax=ax1, orientation='horizontal', label='pdf')
plt.show()

""" Save pdf analysis """

pk.pk_save(filename[:-4] + "_clusters" + str(len(cluster_centers)) +
           "_bw" + str(round(bw, 3)) + ".stat", clusters)

reload = pk.pk_load(filename[:-4] + "_clusters" + str(len(cluster_centers)) +
              "_bw" + str(round(bw, 3)) + ".stat")
