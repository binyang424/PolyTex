"""
input: .coo file
output: .ply file

If interpolate is True, the points farther than the given thereshold (normalzied distance)
will be linearly interpolated to avoid distortion in tetralization process.
"""

import polykriging as pk

path = pk.fileio.choose_directory(titl="coo files")
cwd = pk.fileio.cwd_chdir(path)

filenames = pk.fileio.filenames(path, "coo")

for filename in filenames:
    # extract the yarn number from the filename string using regular expression
    import re
    yarn = int(re.findall(r'\d+', filename)[0])
    print(yarn)
    if yarn in [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
        # coo file to ply file
        pk.coo_to_ply(filename, filename.replace("coo", "ply"), binary=False, interpolate=True, threshold=0.02)
        # pcd file to ply file
        # pk.pcd_to_ply(filename, filename.replace("pcd", "ply"), binary=False)
