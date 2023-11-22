import numpy as np
from tqdm.auto import tqdm
import polykriging as pk
from polykriging.geometry import Plane

##################################################################################
#  Create a textile object
# --------------------------------------------------------------------------------
# 1. create a textile object
# 2. add tows to the textile object
# 3. add groups to the textile object
# 4. remove tows from the textile object. The same tows in self.groups will be
#    removed automatically.
textile = pk.Textile(name="TG96N_Vf57")
print(textile.name)

path = "../Data/22um_Vf57/05_processed_data/transformation/tow/weft/"
files = pk.filenames(path, ".tow")[1:-2]

for file in tqdm(files):
    print(file)
    tow = pk.pk_load(path + file)
    textile.add_tow(tow)

print(textile.tows)
print(textile.tows.keys())

textile.remove("weft_1")  # remove a tow from the textile object
print(textile.tows.keys())

weft_2 = textile.tows["weft_2"]
textile.add_group(name="weft", tow=weft_2)  # add an existing tow to the group
print(textile.groups)

# add an empty group
textile.add_group(name="warp")
print(textile.groups)

tow_new = pk.pk_load(path + files[-1])
print(tow_new.name)
textile.add_group(name="weft", tow=tow_new)  # add a new tow to the group
print(textile.groups)

##################################################################################
#  Create a background mesh for the textile domain
# --------------------------------------------------------------------------------
# 1. define the bounding box of the textile domain
# 2. define the voxel size
# 3. generate the background mesh with textile.mesh()
bbox = np.array((0.6, 12, 1.07, 14.19, 0.15, 5.5))
voxel_size = [0.132, 0.132, 0.066]
textile.mesh(bbox, voxel_size=voxel_size, show=False)