"""
1. the intersection of a curve with a plane
def curve_plane()

2. the intersection of a plane with a plane
def plane_plane()

3. the intersection of a line with a surface
def line_surf()

4. the intersection of a curve with a surface
def curve_surf()

5. the intersection of two surfaces
def surf_surf()
"""

# ！/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def ray_plane_intersect(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
    """
    Find the intersection between a plane and a ray.

    Parameters
    ----------
    plane_normal : array-like
        normal vector of the plane for defining the plane.
    plane_point : array-like
        point on the plane for defining the plane.
    ray_direction : array-like
        direction of the ray.
    ray_point : array-like
        The endpoint of the ray.
    epsilon : float
        tolerance for determining if an intersection point exists.

    Returns
    -------
    Psi: array-like
        intersection point.
    """

    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    ray_direction = np.array(ray_direction)
    ray_point = np.array(ray_point)

    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_point
    return Psi


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """ Make up a test dataset of cylindrical surface """
    m, n = 10, 10  # number of cross-sections and points on each cross-section

    ellipse = generate_elipse_2d(n, 1, 1)

    # ------- generate a grid: n (point label), x, y, z
    pts = np.zeros((m * n, 4))
    for i, z in enumerate(np.linspace(0, 1, m)):
        # number of cross-section
        pts[i * n:(i + 1) * n, 0] = np.arange(n)
        # x, y
        pts[i * n:(i + 1) * n, 1:3] = ellipse
        # z
        pts[i * n:(i + 1) * n, 3] = z

    # ------- take out of the curve for intersection test -------
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')

    """ define the plane for intersection """
    normal = np.array([0.12, 0.01, 1])
    point = np.array([0, 0, 0.5])
    f = plane(normal, point)

    ptIntersects = np.zeros((n, 3))
    for i in range(n):
        mask = pts[:, 0] == i
        curve = pts[mask, 1:]

        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2])

        # find the intersection of the curve with the plane
        ptIntersects[i, :] = find_intersect(f, curve)

    # plot the plane
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x, y = np.meshgrid(x, y)
    zPlane = - f(x, y, x - x) / normal[2]

    ax.plot_surface(x, y, zPlane, alpha=0.4)
    ax.scatter(ptIntersects[:, 0], ptIntersects[:, 1], ptIntersects[:, 2],
               c='r', marker='x', s=60, alpha=0.5)

    """ Plot style settings """
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=15,
                   labelrotation=0, direction='in', length=10, width=1, pad=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=15,
                   labelrotation=0, direction='in', length=5, width=1, pad=0.5)

    plt.show()
