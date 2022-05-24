import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
origin = np.array([0, 0, 0])

# Axis of the elliptical cylinder
p0 = np.array([0, 0, 0])
p1 = np.array([10, 0, 0])

# Radius
R1 = 2; R2 = 1

# Vector in direction of axis
v = p1 - p0
mag = np.linalg.norm(v)     # magnitude of vector
v = v / mag                 #unit vector in direction of axis

# Surface ranges over t from 0 to length of axis and 0 to 2*pi
t = np.linspace(0, mag, 40)

# Generate coordinates for surface
for z in t:
    nPoints = np.random.randint(55, 81)
    theta = np.linspace(0, 2 * np.pi, nPoints)
    
    try:
        xtemp = ( R1 * np.cos(theta) ).reshape([-1,1])
        ytemp = ( R2 * np.sin(theta) ).reshape([-1,1])
        ztemp = np.zeros_like(xtemp)
        ztemp[:] = z
        
        X = np.vstack( ( X, xtemp ) )
        Y = np.vstack( ( Y, ytemp ) )
        Z = np.vstack( ( Z, ztemp ) )
        
    except NameError:
        X = R1 * np.cos(theta).reshape([-1,1])
        Y = R2 * np.sin(theta).reshape([-1,1])
        Z = np.zeros_like(X)
        Z[:] = z


'''  Introduce error  '''
percent = 0.15  # percentage of control points with error

# ± percentage maxDisplacement
maxDisplace = 0.05 # percent

# random error in X and Y direction
rng = default_rng()
numbers = rng.choice(X.size, size= int(percent * X.size), replace=False)
labels = np.sort(numbers)

# Generate random error for X and Y
xerror = R1 * np.hstack( (np.random.rand(int(labels.size/2)) * maxDisplace,
                          - np.random.rand(int(labels.size/2))* maxDisplace ) ) 
np.random.shuffle(xerror)

yerror = R2 * np.hstack( (np.random.rand(int(labels.size/2))* maxDisplace, 
                          - np.random.rand(int(labels.size/2))* maxDisplace ) ) 
np.random.shuffle(xerror)

labels = labels[:xerror.size]

X[labels] = X[labels] + xerror.reshape([-1,1])
Y[labels] = Y[labels] + yerror.reshape([-1,1])

coordinate = np.hstack( (X, Y, Z) )

np.save("./PointsWithError" + str(percent*100) + "Percent.npy", coordinate)

 

import open3d as o3d

# Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector( coordinate )
o3d.io.write_point_cloud("./PointsWithError" + str(percent*100) + "Percent.ply", pcd)


del p0, p1, v, t, R1, R2, rng, xtemp, ytemp, ztemp, z