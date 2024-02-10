"""
Extract surface mesh from image sequence
=========================================
This example shows how to extract a surface mesh from a 3D image sequence
such as a CT scan. The image sequence should be stored as a single tif file.
"""

import polytex as pk
import matplotlib.pyplot as plt
import pandas as pd

#####################################################################
# Load the image sequence
# -----------------------
im = pk.example("image")

mesh, mesh_dim = pk.mesh.im_to_ugrid(im)
mesh.plot(show_edges=True)

#####################################################################
# Get the mesh of fiber tows
# -------------------------
# As we load the image sequence as an unstructured grid, the grey values are
# stored as a point data array. This allows us to extract part of the mesh
# based on the grey value using function pk.mesh.extract_mesh(). It returns
# a volume mesh and a surface mesh of the extracted part.
""" Get the foreground or background mesh """
vol_mesh, surf_mesh = pk.mesh.mesh_extract(mesh, threshold=100, type="foreground")
# vol_mesh.plot(show_edges=True)  # plot the volume mesh
# surf_mesh.plot(show_edges=True)  # plot the surface mesh

#####################################################################
# Separate the mesh according to object connectivity
# --------------------------------------------------
# The extracted mesh may contain multi-tows. We canseparate them according
# to their connectivity using function pk.mesh.mesh_separate() and return
# a dictionary of meshes with the mesh ID as the key.
mesh_dict = pk.mesh.mesh_separation(surf_mesh, plot=False)

# access the fiber tows
binder_0 = mesh_dict["0"]
binder_1 = mesh_dict["1"]

# Plot the surface mesh of tow
binder_0.plot(show_scalar_bar=False, show_edges=False)
binder_1.plot(show_scalar_bar=False, show_edges=False)

#####################################################################
# Reorganize the points of surface mesh in the order of slice (vertical cut plane)
# -------------------------------------------------------------------------------
# The points of the surface mesh are not necessarily well organized. We need to
# reorganize them in the order of slice (vertical cut plane) for further analysis.
points_1_reorder, trajectory = pk.mesh.get_vcut_plane(binder_1, direction='x')

#####################################################################
# Save as point cloud dataset (.pcd)
# ----------------------------------
# We can save the points of the surface mesh as a point cloud dataset (.pcd)
# using function pk.pk_save() for further analysis. The point cloud dataset
# can be loaded by function pk.pk_load().
points_1_df = pd.DataFrame(points_1_reorder, columns=['x', 'y', 'z'])
pk.pk_save(im[:-4] + ".pcd", points_1_df)

#####################################################################
# Visualize the point cloud dataset (.pcd)
# ----------------------------------------
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel('x')
plt.ylabel('y')
# equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

pk.mesh.slice_plot(points_1_reorder, skip=10, marker='o', marker_size=0.1, dpi=300)
