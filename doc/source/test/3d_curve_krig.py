# ！/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D curve kriging
================

This example shows how to use the 3D curve kriging method to interpolate a
3D curve:

.. math:: z = f(x, y)

In this example, we use the following equation to generate a 3D curve for
illustration:

.. math:: x=\sin(t), y= \cos(t), z= \cos(8t)

Note that the function name is possibly been modified in future versions.
"""

import numpy as np
from polykriging.kriging.mdKrig import buildKriging, interp

import matplotlib.pyplot as plt

###############################################################################
# Prepare the data for kriging interpolation
# ------------------
# The data for kriging interpolation should be an array of xy data in shape of
# (n, 2) and an array of z data in shape of (n, 1).
t_rad = np.linspace(0, 2 * np.pi, 10)

xy = np.hstack((np.sin(t_rad).reshape(-1, 1), np.cos(t_rad).reshape(-1, 1)))
z = np.cos(8 * t_rad)

###############################################################################
# Build the kriging model
# -----------------------
# The kriging model is built by the function :func:`buildKriging`. The possible
# drift functions are: ``lin``, ``quad``, and ``cub``, namely, the linear, quadratic
# and cubic drift functions. The default drift function is ``lin``. The possible
# covariance functions are: ``lin``, and ``cub``.
# Nugget effect (nugg) is used for the smoothing of the curve.

expr = buildKriging(xy, z, 'lin', 'cub', nugg=100)

zInterp = interp(xy, expr)

###############################################################################
# Plot the result
# ---------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(xy[:, 0], xy[:, 1], z, '--', label='data')
ax.plot(xy[:, 0], xy[:, 1], zInterp, label='interp/nugg=1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
