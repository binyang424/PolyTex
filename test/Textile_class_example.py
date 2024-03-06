"""
Textile class example
=====================
This example shows how to use the Textile class in PolyTex package. It
is designed to handle the parametrization and geometrical analysis of a fiber Textile. A Tow instance is created by
passing the point cloud of a tow, which consists only the points on the Textile surface, to the constructor."""

import numpy as np
import polytex as ptx

##################################################################################
#  Create a textile object
# ------------------------
# 1. create a textile object
# 2. add tows to the textile object
# 3. add groups to the textile object
# 4. remove tows from the textile object. The same tows in self.groups will be
#    removed automatically.
textile = ptx.Textile(name="TG96N_Vf57")
print(textile.name)

path = "./sample_data/tow/"
files = ptx.filenames(path, ".tow")

for file in files:
    print(path + file)
    tow = ptx.pk_load(path + file)
    textile.add_tow(tow)

print(textile.items)

textile.remove("binder_104")  # remove a tow from the textile object
print(textile.items)

weft_128 = textile["weft_128"]
textile.add_group(name="weft", tow=weft_128)  # add an existing tow to the group
print(textile.groups)

# add an empty group
textile.add_group(name="binder")
print(textile.groups)

# add a tow to the group
textile.add_group(name="binder", tow=textile['binder_105'])  # add a new tow to the group
print(textile.groups)

##################################################################################
# Create a background mesh for the textile domain
# -----------------------------------------------
# 1. define the bounding box of the textile domain
# 2. define the voxel size
# 3. generate the background mesh with textile.mesh()
bbox = np.array((0.6, 12, 1.07, 14.19, 0.15, 5.5))
voxel_size = [0.132, 0.132, 0.066]

textile.meshing(bbox, voxel_size=voxel_size, show=True,
                labeling=True, surface_mesh="./stl/", verbose=False)