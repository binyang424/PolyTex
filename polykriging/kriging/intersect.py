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


def generate_elipse_2d(n, a, b, pc=[0, 0]):
    """
    generate points on an ellipse.
    
    Parameters
    ----------
    n : number of points
    a : semi-major axis
    b : semi-minor axis
    pc : center of the ellipse

    Returns
    -------
    xy: array-like
        Points on the ellipse with shape (n, 3).
    """
    xy = np.zeros((n, 2))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    pc = np.array(pc, dtype=np.float32)
    xy[:, 0] = a * np.cos(theta) + pc[0]
    xy[:, 1] = b * np.sin(theta) + pc[1]

    return xy


def plane(normal, point):
    """
    Define a plane by a normal vector and a point on the plane.

    Note
    ----
        normal = (a, b, c)
        point = (x0, y0, z0)
        Equation of a plane: a(x-x0) + b(y-y0) + c(z-z0) = 0

    Parameters
    ----------
    normal : array-like
        normal vector of the plane.
    point : array-like
        point on the plane.

    Returns
    -------
    f : function
        function of plane.

    Example
    -------
    >>> normal = [0.12, 0.01, 1]
    >>> point = [0, 0, 0.5]
    >>> f = plane(normal, point)
    >>> f
    <function __main__.plane.<locals>.<lambda>(x, y, z)>
    """
    # definition of the plane for intersection with the curve
    normal = np.array(normal)
    point = np.array(point)

    f = lambda x, y, z: normal[0] * (x - point[0]) \
                        + normal[1] * (y - point[1]) \
                        + normal[2] * (z - point[2])
    return f


def find_intersect(f, curve, niterations=5, mSegments=5):
    """
    Find the intersection of a curve with a plane

    Parameters
    ----------
    f : lambda function
        function of plane.
    curve : array-like
        points on the curve in shape of (n, 3).
    niterations: int
        number of iterations.
    mSegments: int
        number of segments for each iteration.

    Returns
    -------
    intersection: array-like
        intersection points with shape (n, 3).
    """
    # check if curve is a numpy array
    if not isinstance(curve, np.ndarray):
        curve = np.array(curve)
    if curve.shape[1] != 3:
        raise RuntimeError("The shape of curve must be (n, 3).")

    fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
    idx = np.where(np.diff(np.sign(fSign)))[0]

    # if there is no intersection
    if len(idx) == 0:
        raise RuntimeError("No intersection was detected.")

    # TODO: cases with more than one intersection
    while niterations > 0:
        # refine the result by linear interpolation
        niterations -= 1
        low = curve[idx, :]
        high = curve[idx + 1, :]
        curve = np.squeeze(
            np.linspace(low, high, mSegments).reshape(1, -1, 3))
        fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
        idx = np.where(np.diff(np.sign(fSign)))[0]

    intersect = (curve[idx, :] + curve[idx + 1, :]) / 2

    return intersect


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

    elipse = generate_elipse_2d(n, 1, 1)

    # ------- generate a grid: n (point label), x, y, z
    pts = np.zeros((m * n, 4))
    for i, z in enumerate(np.linspace(0, 1, m)):
        # number of cross-section
        pts[i * n:(i + 1) * n, 0] = np.arange(n)
        # x, y
        pts[i * n:(i + 1) * n, 1:3] = elipse
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
