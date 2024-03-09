"""
Tow class example
=================
This example shows how to use the Tow class in PolyTex package. It is designed to handle the parametrization and geometrical analysis of a fiber tow. A Tow instance is created by passing the point cloud of a tow, which consists only the points on the tow surface, to the constructor.
"""

#####################################################################
# Example dataset
# ---------------
import polytex as ptx
import numpy as np

# Load the surface points of fiber tow
path = ptx.example("surface points")
surf_points = ptx.read_explicit_data(path)

resolution = 0.022  # mm/pixel
surf_points = surf_points * resolution  # convert to millimeters

#####################################################################
# We clip the coordinates to discard the part of the tow that is necessary for
# the modeling of the tow. Optional.
mask = (surf_points[:, 0] > 1.1 - 0.2) & (surf_points[:, 0] < 11.55 + 0.2)
surf_points_clip = surf_points[mask]

#####################################################################
# Exchange the first and last column for geometry analysis. Note that
# the last column is deemed as the label column. Namely, the points
# with the same label are considered as belonging to the same slice and
# are parametrized in the radial direction (theta). This data reorder
# is necessary for the users.
coordinates = surf_points_clip[:, [0, 1, 2]]

#####################################################################
# We can filtering the points to remove noise or the points that are not necessary:
# mask = abs(coordinates[:, -1] - 9.196) > 0.09
# coordinates = coordinates[mask]

#####################################################################
# Create a Tow instance with PolyTex Tow class
# --------------------------------------------
tow = ptx.Tow(
        surf_points=coordinates,
        order="xyz",
        rho_fiber=2550,
        radius_fiber=6.5e-6,
        length_scale="mm", tex=1100,
        name="weft_0")

#####################################################################
# Get the parametric coordinates of the tow: The points on the same slice
# are parametrized in the radial direction (theta) and stored in the
# normalized distance column of the attribute, tow.coordinates, a pandas
# DataFrame.
df_coord = tow.coordinates  # parametric coordinates of the tow

#####################################################################
# Get the geometrical features of the tow: The geometrical features of the tow
# are stored in the attribute, tow.geom_features, a pandas DataFrame.
# For straight fiber tows, the geometrical features can be used as an approximation
# of the actual tow geometry. But for wavy tows, such as binder, the geometrical
# features are not accurate enough. We need to redo the geometrical analysis
# after identifying the normal cross-sections of the tow.
df_geom = tow.geom_features  # geometrical features of the tow

################################################################################
# Resampling
# ----------
# Resampling the control points of the tow with a uniform spacing in the
# normalized distance direction. The resampling is necessary to create a
# parametric representation based on dual kriging.

# # Equidistant resampling of the tow control points in the radial direction.
# theta_res = 35  # number of control points in the radial direction
# sample_position = np.linspace(0, 1, theta_res, endpoint=True)  # equal spaced points (normalized distance)

# # Resampling according to distribution density
cluster = tow.kde(bw=0.01)
sample_position = cluster["cluster centers"]

pts_krig, expr_krig = tow.resampling(krig_config=("lin", "cub"),
                                     skip=10, sample_position=sample_position,
                                     smooth=0.0001)

#####################################################################
# Save and reload the tow instance
# --------------------------------
# tow.save("./tow/weft_0.tow")

#####################################################################
# Plot the tow
# ------------
mesh = tow.surf_mesh(plot=True, save_path="./test_data/weft_0.ply", end_closed=True)

#####################################################################
# Smooth the tow trajectory with Kriging
# --------------------------------------
trajectory_sm = tow.trajectory(smooth=0.0015, plot=False,
                               save_path="./test_data/trajectory.ply", orientation=True)

#####################################################################
# Axial and radial lines
# ----------------------
# Get the axial lines of the tow (the lines connecting the parametrized control points in
# the axial direction)
line_axi = tow.axial_lines(plot=True)

# Get the radial lines of the tow (the lines connecting the parametrized control points in
# the radial direction)
line_rad = tow.radial_lines(plot=True)

################################################################################
# Get the normal cross-sections of the tow
# ----------------------------------------
# So far, we provide two methods to get the normal cross-sections of the tow.
# The first method wraps the intersection function of plane and surface mesh
# in the pyvista package. The second method is based on the intersection of
# a parametric curve and an implicit plane.
cross_section, planes, clipped = tow.normal_cross_section(algorithm="pyvista")

#####################################################################
# Update the geometrical features of the tow
# ------------------------------------------
# The geometrical features of the tow are stored in the attribute, tow.geom_features, a pandas DataFrame.
# You have this information once the tow instance is created. However, that is calculated based on the vertical
# cross-sections of the tow. A more accurate geometrical analysis can be done during the identification of
# the normal cross-sections with the class method, Tow.normal_cross_section.
# Acess the updated geometry features according to the normal cross-sections.
df_geom_pv = tow.geom_features.copy()

################################################################################
# Get the normal cross-sections of the tow
# ----------------------------------------
# The kriging method is based on the intersection of a parametric curve and an implicit plane.
cross_section, plane, clipped = tow.normal_cross_section(algorithm="kriging", plot=True,
                                                             i_size=2, j_size=3, skip=15)
# Acess the updated geometry features according to the normal cross-sections.
df_geom_krig = tow.geom_features

################################################################################
# Geometry features
# -----------------
# as shwon above, the tow geometry features can be updated after the normal cross-sections
# are identified using both method. However, the accuracy are different. The pyvista method
# is faster but less accurate. In the kriging method, we transform the identified cross-sections
# to a 2d plane. The geometry features are then calculated based on the 2d coordinates. Thus, the
# geometry features are more accurate than the pyvista method. However, this also makes the
# kriging method less efficient. The kriging method is recommended for wavy tows, such as binder.
