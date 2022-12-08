"""
Coordinate transformation
=========================
The following code is a demonstration of coordinate transformation using direct
cosine matrix (DCM) and Euler angles (phi, theta, psi).
"""
import polykriging as pk
from polykriging.geometry import transform as tf
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

# ============================================================
# Load test data
path = pk.example(data_name="cross-section")
data = pv.read(path)
normal = [0.43583834, -0.00777955, -0.89999134]
data.plot()

# ============================================================
# Get the data points and its centroid
points = data.points
centroid = np.mean(points, axis=0)

# ============================================================
# Move the centroid to the origin of global coordinate system
points = points - centroid

# ============================================================
# We want to rotate the global coordinate system to align its z-axis
# with the normal vector of the cross-section. To do this, we need to
# find the euler angles (phi, theta, psi).
angles = tf.euler_z_noraml(normal)

# ============================================================
# We can use the euler angles to construct the DCM
dcm = tf.e123_dcm(*angles)

# ============================================================
# Rotate the points
points = np.dot(dcm, points.T).T

# ============================================================
# Plot the rotated points
plt.plot(points[:, 0], points[:, 1], "o")
# equal aspect ratio
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# ============================================================
# Note: The points have to be ordered in a clockwise or
# counter-clockwise manner. So we need to sort the points first
# before we can use them to construct a polygon and find the area
# or perimeter.
