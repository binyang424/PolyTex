"""
Coordinate transformation
=========================
The following code is a demonstration of coordinate transformation using direct
cosine matrix (DCM) and Euler angles (phi, theta, psi).
"""
import polytex as pk
from polytex.geometry import transform as tf
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

#############################################################
# Load test data
# --------------
path = pk.example(data_name="cross-section")
data = pv.read(path)
normal = [0.43583834, -0.00777955, -0.89999134]
data.plot()

#############################################################
# Get the data points and its centroid
# ------------------------------------
# The example data is a cross-section of a woven fabric stored in a vtk file.
# This can be loaded using pyvista as shown above. Now we get the points and
# the centroid of the cross-section. We will use the centroid as the origin
# of the local coordinate system.
points = data.points
centroid = np.mean(points, axis=0)

#############################################################
# Translation
# -----------
# Move the centroid to the origin of global coordinate system.
# Note that we did not import the translation function so far.
# So the user should translate the local coordinate system to
# the origin before calling the rotation functions.
points = points - centroid

#############################################################
# Euler angles for rotation
# -------------------------
# We want to rotate the global coordinate system to align its z-axis
# with the normal vector of the cross-section. To do this, we need to
# find the euler angles (phi, theta, psi).
angles = tf.euler_z_noraml(normal)

#############################################################
# Direct cosine matrix
# --------------------
# Now we can use the euler angles to construct the DCM:
dcm = tf.e123_dcm(*angles)

#############################################################
# Check the result
# ----------------
# Rotate the points
points1 = np.dot(dcm, points.T).T

# Plot the rotated points
plt.plot(points1[:, 0], points1[:, 1], "o")
# equal aspect ratio
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

#############################################################
# Notes
# -----
# We need to sort the points first before using them to
# construct a polygon and find the area or perimeter.


#############################################################
# Align the old coordinate system with a new one
# ----------------------------------------------
# Above we aligned the z-axis of the global coordinate system
# with a given vector but without considering the x- and y-axes.
# Now we want to align the old coordinate system with a new one
# defined by two vectors: z_new and x_new. The following code
# shows how to do this.

x_new = points[np.argmax(np.linalg.norm(points, axis=1))]

angles2 = tf.euler_zx_coordinate(normal, x_new)
dcm = tf.e123_dcm(*angles2)
points2 = np.dot(dcm, points.T).T

#############################################################
# Check the result
# ----------------
# Plot the rotated points
plt.plot(points2[:, 0], points2[:, 1], "o")
plt.scatter(0, 0, c="r")
# equal aspect ratio
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

#############################################################
# Compare
# -------
# A comparison between the direct cosine matrix and the coordinate
# the basis vectors of the original coordinate system after rotation.
basis = np.eye(3)
print(np.dot(dcm, basis.T).T)
print(dcm)

