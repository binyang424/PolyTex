"""
Convert a 3D image sequence (implicit) to an explicit dataset
==============================================================
This example shows how to extract a surface mesh from a 3D image sequence
such as a CT scan. The image sequence should be stored as a single tif file.
"""

import polytex as ptx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import os

#####################################################################
# Load the image sequence
# -----------------------
im = ptx.example("image")

# Render the surface of the model
mesh, mesh_dim = ptx.mesh.im_to_ugrid(im)
mesh.plot(show_edges=False, opacity=1)

# # Render as a volume. Note that it can take a while as the mesh size is large.
# vol_pl = pv.Plotter()
# vol_pl.add_volume(mesh, cmap="bone", opacity="sigmoid")
# vol_pl.show()

#####################################################################
# Get the mesh of fiber tows
# --------------------------
# As we load the image sequence as an unstructured grid, the grey values are
# stored as a point data array. This allows us to extract part of the mesh
# based on the grey value using function ptx.mesh.extract_mesh(). It returns
# a volume mesh and a surface mesh of the extracted part.
""" Get the foreground or background mesh """
vol_mesh, surf_mesh = ptx.mesh.mesh_extract(mesh, threshold=100, type="foreground")
vol_mesh.plot(show_edges=True)  # plot the volume mesh
surf_mesh.plot(show_edges=True)  # plot the surface mesh

#####################################################################
# Separate the mesh according to object connectivity
# --------------------------------------------------
# The extracted mesh may contain multi-tows. We canseparate them according
# to their connectivity using function ptx.mesh.mesh_separate() and return
# a dictionary of meshes with the mesh ID as the key.
mesh_dict = ptx.mesh.mesh_separation(surf_mesh, plot=True)

# access the fiber tows
binder_0 = mesh_dict["0"]
binder_1 = mesh_dict["1"]

# Plot the surface mesh of tow seperately
binder_0.plot(show_scalar_bar=False, show_edges=False)
binder_1.plot(show_scalar_bar=False, show_edges=False)

# Or plot the surface mesh of each fiber tow with one window:
pl = pv.Plotter()
for binder in [binder_0, binder_1]:
    pl.add_mesh(binder, show_edges=False)
    print(binder.bounds)
pl.show()

#####################################################################
# Reorganize the points of surface mesh in the order of slice (vertical cut plane)
# --------------------------------------------------------------------------------
# The points of the surface mesh are not necessarily well organized. We need to
# reorganize them in the order of slice (vertical cut plane) for further analysis.
points_0_reorder, trajectory_0 = ptx.mesh.get_vcut_plane(binder_0, direction='x')
points_1_reorder, trajectory_1 = ptx.mesh.get_vcut_plane(binder_1, direction='x')

# Plot the trajectories of identified fiber tows
for traj in [trajectory_0, trajectory_1]:
    plt.plot(traj[:, 0], traj[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    # equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

ptx.mesh.slice_plot(points_0_reorder, skip=4, marker='o', marker_size=0.1, dpi=300)
ptx.mesh.slice_plot(points_1_reorder, skip=4, marker='o', marker_size=0.1, dpi=300)

points_0_reorder = points_0_reorder[:, [2, 1, 0]]  # z, y, x
points_1_reorder = points_1_reorder[:, [2, 1, 0]]

#####################################################################
# Export as explicit dataset
# --------------------------
# Operations above convert an implicit dataset (image masks) into an
# explicit dataset (point clouds describing the surface of fiber tows).
# We can export the point clouds to a CSV file for further analysis
# with the built-in functions of PolyTex `ptx.read_explicit_data()`.

slice_ind = np.arange(0, 276, 1)

if not os.path.exists("./test_data/csv/"):
    os.makedirs("./test_data/csv/")

for i in range(slice_ind.size):
    pts_slice = points_0_reorder[points_0_reorder[:, 2] == slice_ind[i], :]
    if pts_slice.size != 0:
        pts_slice_df = pd.DataFrame(pts_slice, columns=['x', 'y', 'z'])
        pts_slice_df.to_csv("./test_data/csv/binder_0_" + str(i) + ".csv", index=True)

    pts_slice = points_1_reorder[points_1_reorder[:, 2] == slice_ind[i], :]
    if pts_slice.size != 0:
        pts_slice_df = pd.DataFrame(pts_slice, columns=['x', 'y', 'z'])
        pts_slice_df.to_csv("./test_data/csv/binder_1_" + str(i) + ".csv", index=True)