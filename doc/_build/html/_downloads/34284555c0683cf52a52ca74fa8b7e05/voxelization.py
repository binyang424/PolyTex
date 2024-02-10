"""
Voxelization of surface mesh
============================================================

Test

"""

import numpy as np
import pyvista as pv
from polytex.mesh import features

# voxelize the mesh
mesh = pv.read("./testdata/profile_0_weft.stl")

# test the effect of density on mesh volume and cell size
density = [0.088, 0.11, 0.088]
density = 0.022

vox1, ugrid = features.voxelize(mesh, density=density, density_type='cell_size', contained_cells=False)

cell_centers = pv.wrap(ugrid.cell_centers().points)
selection = cell_centers.select_enclosed_points(
    mesh.extract_surface(), tolerance=0.0, check_surface=True
)
mask = selection.point_data['SelectedPoints'].view(np.bool_)
vox2 = ugrid.extract_cells(mask)

# save mesh vox2
vox2.plot(show_edges=True)
vox2.save("./testdata/profile_0_weft_vox2.vtk")

# vox3, _ = features.voxelize(mesh, density=density, density_type='cell_size', contained_cells=True)
#
# pv.global_theme.font.size = 10
# pv.set_plot_theme("paraview")
# pv.global_theme.font.family = 'times'
# pl = pv.Plotter(shape=(2, 2))
# # pl.set_background("white", top="white")
#
# pl.subplot(0, 0)
# _ = pl.add_mesh(mesh, show_edges=True)
# _ = pl.add_title('Surface Mesh \n volume = {}'.format(round(mesh.volume, 3)))
#
# pl.subplot(0, 1)
# _ = pl.add_mesh(vox1, show_edges=True)
# _ = pl.add_title('Any vertices \n volume = {}'.format(round(vox1.volume, 3)))
#
# pl.subplot(1, 0)
# _ = pl.add_mesh(vox2, show_edges=True)
# _ = pl.add_title('Cell Center contained \n volume = {}'.format(round(vox2.volume, 3)))
#
# pl.subplot(1, 1)
# _ = pl.add_mesh(vox3, show_edges=True)
# _ = pl.add_title('All vertices \n volume = {}'.format(round(vox3.volume, 3)))
#
# pl.show()

# 0.022: 11.63769965596028; 10.728850263964102; 9.750362951967027
# 0.044: 12.513; 10.726; 8.757
# 0.066: 13.635; 10.729; 8.038
# 0.088: 14.27; 10.655; 6.884
# 0.11: 15.248; 10.723; 6.173
# 0.132: 16.143; 10.897; 5.322
# 0.154: 16.684; 10.862; 4.313
# 0.176: 17.429; 10.947; 3.2
# 0.198: 19.15; 10.922; 3.454
# 0.22: 19.166; 9.828; 3.003

# voxel size and mesh volume and vox1, vox2, and vox3 volume, respectively.
volumes = np.array([[0.022, 11.638, 10.729, 9.750],
                    [0.044, 12.513, 10.726, 8.757],
                    [0.066, 13.635, 10.729, 8.038],
                    [0.088, 14.27, 10.655, 6.884],
                    [0.11, 15.248, 10.723, 6.173],
                    [0.132, 16.143, 10.897, 5.322],
                    [0.154, 16.684, 10.862, 4.313],
                    [0.176, 17.429, 10.947, 3.2],
                    [0.198, 19.15, 10.922, 3.454],
                    [0.22, 19.166, 9.828, 3.003]])

import matplotlib.pyplot as plt

# font size and font family
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

# plot a horizontal line at the volume of the mesh
plt.plot([0.02, 0.22], [10.735, 10.735], 'k--')

plt.plot(volumes[:, 0], volumes[:, 1], 'o-')
plt.plot(volumes[:, 0], volumes[:, 2], 'x-')
plt.plot(volumes[:, 0], volumes[:, 3], '*-')

plt.legend(['Actual volume', 'Vertices based', 'Centroid based', 'Cell based'])
plt.xlabel('Voxel size ($mm$)')
# specified x-axis tick labels
plt.xticks([0.022, 0.044, 0.066, 0.088, 0.11, 0.132, 0.154, 0.176, 0.198, 0.22])
plt.ylabel(r'Tow volume ($mm^3$)')
# tight layout
plt.tight_layout()
plt.savefig('./testdata/voxelization.png', dpi=600)
