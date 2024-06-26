"""
Mesh labelling and intersection detection
=========================================

Test

"""

import re
import time

import numpy as np
import pyvista as pv
from polytex.io import save_nrrd, choose_directory, filenames
from polytex.mesh import background_mesh, label_mask, intersection_detect
from scipy.sparse import coo_matrix

""" Inputs """
# Generate a voxel background mesh.
bbox = np.array((0.0, 12.21, 0.5, 10.4, 0.20, 5.37))
voxel_size = [0.11, 0.11, 0.11]
yarnIndex = np.arange(0, 52)

""" Generate a voxel background mesh. """
mesh_background, mesh_shape = background_mesh(bbox, voxel_size)

""" Plot the background mesh. """
mesh_background.plot(show_edges=True, color='w', opacity=1)
# mesh.save("./file/test_bbox.vtu", binary=True)

# time labelling
start_labelling = time.time()

# initialize the array of label list with -1
label_list = np.full(mesh_background.n_cells, -1, dtype=np.int32)

""" Select the surface meshes of yarns to be labelled """
label_set_dict = dict()
path = choose_directory("Choose the surface mesh directory for fiber tow labelling")
file_list = filenames(path, ".stl")
file_list_sort = {}
for i, file in enumerate(file_list):
    # regular expression for integer
    yarn_index = re.findall(r'\d+', file)
    file_list_sort[int(yarn_index[0])] = file

""" Label the surface meshes of yarns """
indices = np.array(list(file_list_sort.keys()))
indices.sort()
for index in indices:
    if index in [47, 48, 49]:
        continue

    print("Processing yarn %d of %d" % (index, len(yarnIndex)))

    mesh_tri = pv.read(path + "\\" + file_list_sort[index])  # load surface mesh

    # find the cells that are within the tubular surface of the fiber tow
    mask, label_yarn = label_mask(mesh_background, mesh_tri)

    label_list[mask] = index
    label_set_dict[index] = coo_matrix(label_yarn)

print("Labelling time: %.2f seconds" % (time.time() - start_labelling))

""" Find the intersection of fiber tows  """
intersect_info, intersect_info_dict, cell_data_intersect = intersection_detect(label_set_dict)

mesh_background.cell_data['label'] = label_list
mesh_background.cell_data['intersection'] = cell_data_intersect
# mesh_background.save('./file/test_bbox_cells.vtu', binary=True)
save_nrrd(label_list, mesh_shape, "./file/test_bbox_cells")
