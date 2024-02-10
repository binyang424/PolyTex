"""
2D curve kriging with confidence
================================

This example shows how to interpolate a 2D curve with confidence estimation.

.. math:: y = f(x)

where :math:`f` is a 2D curve. The curve is defined by a set of points
:math:`(x_i, y_i)`, where :math:`i = 1, 2, ..., n`.

This kriging method is the basis for fiber tow trajectory smoothing and control
point resampling of fiber tow surface implimented in polytex.Tow class.
"""

import numpy as np
from polytex.kriging import curve2D
import polytex as pk
import matplotlib.pyplot as plt

# Make up some data
X = np.linspace(start=0, stop=10, num=300)
y = X * np.sin(X)

# Choose some data points randomly for training
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=16, replace=False)

X_train, y_train = X[training_indices], y[training_indices]

data_set = np.hstack((X_train.reshape(-1, 1), y_train.reshape(-1, 1)))

name_drift, name_cov = 'lin', 'cub'
nuggetEffect = 0

##############################################################################
# Matrice and vectors for dual Kriging formulation
# ------------------------------------------------
# For most users, this part can be ignored. It is only for the purpose of
# understanding the formulation of dual Kriging. Kriging interpolation can be
# achieved by calling the function ``curve2D.curve2Dinter``
mat_krig, mat_krig_inv, vector_ba, expr, func_drift, func_cov = \
    curve2D.curveKrig1D(data_set, name_drift, name_cov, nuggetEffect=nuggetEffect)

##############################################################################
# Kriging interpolation
# ---------------------
# Kriging model and prediction with mean, Kriging expression
# and the corresponding standard deviation as output.
mean_prediction, expr, std_prediction = curve2D.curve2Dinter(
    data_set, name_drift, name_cov,
    nuggetEffect=nuggetEffect, interp=X, return_std=True)

# Plot the results
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations", s=50, zorder=10)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(X.ravel(),
                 mean_prediction - 1.96 * std_prediction,
                 mean_prediction + 1.96 * std_prediction,
                 alpha=0.5, label=r"95% confidence interval")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("2D curve Kriging regression on noise-free dataset")

plt.show()

##############################################################################
# .. image:: images/2D_curve_kriging_with_confidence.png

##############################################################################
# Save the Kriging model
# ----------------------
# You can save the Kriging model to a file for later use and load it back
# using pk.load() function. Note that the Kriging model is saved in a Python
# dictionary with its name as the key.
expr_dict = {"cross": expr}
pk.pk_save("./test_data/FunXY.krig", expr_dict)
expr_load = pk.pk_load("./test_data/FunXY.krig")