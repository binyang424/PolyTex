import numpy as np
import pyvista as pv
from polykriging import features
import nrrd

# create a voxelized cube
center = (40, 40, 40)
cube = pv.Cube(center=center, x_length=81, y_length=81, z_length=81)

density = 1

cub_vox, ugrid = features.voxelize(cube, density=density, density_type='cell_size',
                                   contained_cells=False)
cub_vox.plot(show_edges=True)
cell_centers = np.array(cub_vox.cell_centers().points)

indicator = np.zeros(cell_centers.shape[0])
for i in range(cell_centers.shape[0]):
    if np.linalg.norm(cell_centers[i, :] - center) < 29:
        indicator[i] = 1

header = {'type': 'uint8', 'dimension': 3, 'space': 'left-posterior-superior',
            'space directions': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'sizes': [80, 80, 80], 'kinds': ['domain', 'domain', 'domain'],
            'encoding': 'gzip', 'endian': 'little', 'space origin': [0, 0, 0]}


nrrd.write('./testdata/cube_vox.nrrd', indicator.reshape((80, 80, 80)[:,:, ]),
           header=header, index_order='C')