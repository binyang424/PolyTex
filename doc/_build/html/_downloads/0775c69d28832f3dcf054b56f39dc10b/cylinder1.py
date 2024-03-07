"""
Test
=================

Test

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from polykriging import surf2stl

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
origin = np.array([0, 0, 0])

# Axis of the elliptical cylinder
p0 = np.array([0, 0, 0])
p1 = np.array([10, 0, 0])

# Radius
R1 = 2
R2 = 1

# Vector in direction of axis
v = p1 - p0
mag = np.linalg.norm(v)     # magnitude of vector
v = v / mag                 #unit vector in direction of axis

# Make some vector not in the same direction as v
not_v = np.array([1, 0, 0])
if (v == not_v).all():
    not_v = np.array([0, 1, 0])

# Make vector perpendicular to v
n1 = np.cross(v, not_v)
n1 /= np.linalg.norm(n1)    #normalize n1

# Make unit vector perpendicular to v and n1
n2 = np.cross(v, n1)


# Surface ranges over t from 0 to length of axis and 0 to 2*pi
t = np.linspace(0, mag, 40)
theta = np.linspace(0, 2 * np.pi, 50)
t, theta = np.meshgrid(t, theta)    #use meshgrid to make 2d arrays


# Generate coordinates for surface
X, Y, Z = [p0[i] + v[i] * t + R1 * np.sin(theta) * n1[i] + R2* np.cos(theta) * n2[i] for i in [0, 1, 2]]


T, Theta = t.flatten(), theta.flatten()

X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()


delaunay_tri = Delaunay(np.array([T, Theta]).T)
surf2stl.tri_write('mobius.stl', X, Y, Z, delaunay_tri)

##ax.plot_surface(X, Y, Z)
###plot axis
##ax.plot(*zip(p0, p1), color = 'red')
####ax.set_xlim(0, 10)
####ax.set_ylim(0, 10)
####ax.set_zlim(0, 10)
##plt.show()
