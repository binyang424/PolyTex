"""
Tubular mesh with functions in polytex.mesh
================================================
The fiber tow surface can be regarded as a tubular structure. Thus, it is
important to construct a tubular mesh for further analysis.

This example shows how to create a tubular mesh with constant cross-section. The
cross-section is defined by a set of ellipse points. The parameters of the
ellipse are the major and minor axis, and the number of points on the ellipse.

Note that we already implemented a primitive geometry generator in
polytex.geometry.Tube class. It is recommended to use that class to
generate tubular mesh. This example is only for demonstration purpose.
"""

import polytex.mesh as ms
import meshio

theta_res = 5
h_res = 5
h = 15
a = 4
b = 1

###############################################################################
# Generate the tubular mesh vertices
# ----------------------------------
points = ms.structured_cylinder_vertices(a=a, b=b, h=h,
                                         theta_res=theta_res, h_res=h_res)

###############################################################################
# Generate the tubular mesh cells
# -------------------------------
mesh = ms.tubular_mesh_generator(theta_res=theta_res,
                                 h_res=h_res,
                                 vertices=points)
mesh.plot(show_edges=True)
###############################################################################
# Extract information from the mesh object
# --------------------------------------------
points, cells, point_data, cell_data = ms.to_meshio_data(mesh,
                                                         theta_res,
                                                         correction=True)
###############################################################################
# Write the mesh to a file with meshio
# ------------------------------------
meshio.write_points_cells(
    filename="cylinder.ply",
    points=points,
    cells=cells, binary=False
)

