"""
Convert the TexGen-generated vtu file to OpenFOAM polyMesh
==========================================================

This example demonstrates how to convert the TexGen-generated vtu file to OpenFOAM
polyMesh. The cell data of the vtu file will be written to the OpenFOAM mesh. The
boundary type for the OpenFOAM mesh is also defined. A OpenFOAM case template is
provided by polytex to prepare the case for OpenFOAM.
"""


import os
import pyvista as pv
import polytex as ptx


# input parameters
rf = 8.5e-6  # m, fiber radius
fp = "texgen_50_50_50.vtu"   # file path
scale = 1e-3  # scale the mesh from mm to m

mesh = pv.read(fp)
mesh = ptx.texgen_voxel(mesh, rf, plot=True)

""" 
Mesh writing 
============
"""
output_dir = "./foam_case/"
# the cell data to be written to the OpenFOAM mesh
cell_data = {"K", "D", "VolumeFraction", "YarnIndex", "Orientation"}
# the boundary type for the OpenFOAM mesh
boundary_type = {"left": "wall", "right": "wall", "front": "patch", "back": "patch",
                 "bottom": "wall", "top": "wall"}
# write the mesh to OpenFOAM polyMesh
ptx.voxel2foam(mesh, scale=scale, outputDir=output_dir, boundary_type=boundary_type,
              cell_data_list=cell_data)

""" Create a OpenFOAM case file for Paraveiw post-processing """
with open(os.path.join(output_dir, "test.foam"), "w") as f:
    pass  # empty file
f.close()

""" Prepare the case for OpenFOAM with the template provided by polytex """
ptx.case_prepare(output_dir)
print("Case preparation is done!")