import polykriging.mesh as ms
import meshio
import numpy as np

theta_res = 5
h_res = 5
h = 15
a = 4
b = 1

points = ms.structured_cylinder_vertices(a=a, b=b, h=h,
                                         theta_res=theta_res, h_res=h_res)
mesh = ms.tubular_mesh_generator(theta_res=theta_res,
                                 h_res=h_res, vertices=points)

points, cells, point_data, cell_data = ms.to_meshio_data(mesh, theta_res, correction=True)


meshio.write_points_cells(
    filename="cylinder.ply",
    points=points,
    cells=cells, binary=False
)

