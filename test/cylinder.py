import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
from scipy.spatial import Delaunay
from polykriging import surf2stl

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
origin = np.array([0, 0, 0])

#axis and radius
p0 = np.array([1, 3, 5])
p1 = np.array([8, 3, 5])
R1 = 5
R2 = 3

#vector in direction of axis
v = p1 - p0
#find magnitude of vector
mag = norm(v)
#unit vector in direction of axis
v = v / mag
#make some vector not in the same direction as v
not_v = np.array([1, 0, 0])
if (v == not_v).all():
    not_v = np.array([0, 1, 0])
#make vector perpendicular to v
n1 = np.cross(v, not_v)
#normalize n1
n1 /= norm(n1)
#make unit vector perpendicular to v and n1
n2 = np.cross(v, n1)
#surface ranges over t from 0 to length of axis and 0 to 2*pi
t = np.linspace(0, 2*mag, 50)
theta = np.linspace(0, 2 * np.pi, 50)
#use meshgrid to make 2d arrays
t, theta = np.meshgrid(t, theta)
#generate coordinates for surface
X, Y, Z = [p0[i] + v[i] * t + R1 * np.sin(theta) * n1[i] + R2* np.cos(theta) * n2[i] for i in [0, 1, 2]]




u, v = t.flatten(), theta.flatten()

X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()


delaunay_tri = Delaunay(np.array([u, v]).T)
surf2stl.tri_write('mobius.stl', X, Y, Z, delaunay_tri)

##ax.plot_surface(X, Y, Z)
###plot axis
##ax.plot(*zip(p0, p1), color = 'red')
####ax.set_xlim(0, 10)
####ax.set_ylim(0, 10)
####ax.set_zlim(0, 10)
##plt.show()
