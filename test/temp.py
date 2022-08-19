# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='3d')


def generate_points(n, a, b):
    """
    generate points on an ellipse
    :param n: number of points
    :param a: semi-major axis
    :param b: semi-minor axis
    :return:
    """
    data = np.zeros((n, 4))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    data[:, 0] = np.arange(0, n)
    data[:, 1] = a * np.cos(theta)
    data[:, 2] = b * np.sin(theta)
    return data


def find_intersect(f, curve, niterations=5, mSegments=5):
    """
    find the intersection of a curve with a plane
    :param f: function of the plane
    :param curve:
    :param niterations:
    :param mSegments:
    :return:
    """
    fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
    idx = np.where(np.diff(np.sign(fSign)))[0]

    while niterations > 0:
        niterations -= 1
        low = curve[idx, :]
        high = curve[idx + 1, :]
        curve = np.squeeze(np.linspace(low, high,
                                       mSegments).reshape(1, -1, 3))
        fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
        idx = np.where(np.diff(np.sign(fSign)))[0]
    return (curve[idx, :] + curve[idx + 1, :]) / 2


m, n = 10, 10  # number of cross-sections and points on each cross-section

data = generate_points(n, 1, 1)

# generate a grid: n (point label), x, y, z
pts = np.zeros((m * n, 4))
for i, z in enumerate(np.linspace(0, 1, m)):
    data[:, -1] = z
    pts[i * n:(i + 1) * n, :] = data

# definition of the plane for intersection
f = lambda x, y, z: 0.12 * x + 0.01 * y + 0.5 - z

data[:, 3] = 0
zIntersect = f(data[:, 1], data[:, 2], data[:, 3])

# ------- take out of the line -------
ptIntersects = np.zeros((n, 3))
for i in range(n):
    mask = pts[:, 0] == i
    curve = pts[mask, 1:]

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])
    ptIntersects[i,:] = find_intersect(f, curve)

# # plot
# ax.scatter(pts[:, 1], pts[:, 2], pts[:, 3], c='k',
#            marker='o', s=3, alpha=0.5)

# plot a plane
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
zPlane = f(x, y, x-x)

ax.plot_surface(x, y, zPlane, alpha=0.4)
ax.scatter(ptIntersects[:, 0], ptIntersects[:, 1], ptIntersects[:, 2],
              c='r', marker='x', s=60, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
