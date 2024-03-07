"""
Step 2: Convert .coo file to .ply for point cloud visualization
===============================================================
input: .coo file
output: .ply file

If interpolate is True, the points farther than the given threshold (normalzied distance) will be linearly interpolated to avoid distortion in tetralization  process.
"""

import polytex as pk

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

path = pk.io.choose_directory(titl="coo files")
cwd = pk.io.cwd_chdir(path)

filenames = pk.io.filenames(path, "coo")

for filename in filenames:
    # extract the yarn number from the filename string using regular expression
    import re
    yarn = int(re.findall(r'\d+', filename)[0])
    print(yarn)
    if yarn in range(0, 50):
        # coo file to ply file
        pk.coo_to_ply(filename, filename.replace("coo", "ply"), binary=False, interpolate=True, threshold=0.02)
        # pcd file to ply file
        # pk.pcd_to_ply(filename, filename.replace("pcd", "ply"), binary=False)
