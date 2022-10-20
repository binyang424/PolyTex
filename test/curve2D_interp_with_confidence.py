import numpy as np
from polykriging.kriging import curve2D
import polykriging as pk
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

# # Matrice and vectors for dual Kriging formulation
# mat_krig, mat_krig_inv, vector_ba, expr, func_drift, func_cov = \
#     curve2D.curveKrig1D(data_set, name_drift, name_cov, nuggetEffect=nuggetEffect)

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

expr_dict = {"cross": expr}
pk.pk_save("FunXY.krig", expr_dict)
expr_load = pk.pk_load("FunXY.krig")
