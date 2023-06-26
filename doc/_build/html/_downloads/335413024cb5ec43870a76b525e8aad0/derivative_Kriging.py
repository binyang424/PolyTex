"""
Derivative kriging
==================

This example shows how to use of derivative kriging for interpolation y = f(x)
with given derivative dy/dx = f'(x) in some position.
"""

import matplotlib.pyplot as plt
from polykriging import kriging

#####################################################################
# Example 1
# ---------
# Make up a test dataset

x = [0, 2.5, 4, 4.5, 5.08]
y = [5, 8, 9, 5, -3]

# Note that the derivative is given as a list of (dy/dx) corresponding to x that
# is also stored in a list.
x_deriv = [0]
y_deriv = [-1]

# define the kriging model
sum_ave = kriging.bd_Deriv_kriging_func(x, y, x_deriv, y_deriv, 'cst', 'cub', 100, 0)

print(sum_ave)

plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='upper left', ncol=1)

plt.clf() # clear matplotlib figure

#####################################################################
# Example 2
# ---------
# Make up a test dataset
x = [0, 1]
y = [0, 0]

x_deriv = [0.5]
y_deriv = [1]

# bd_Deriv_kriging_func(x, y, xDeriv, yDeriv, choixDerive, choixCov, plot_x_pts, nugg)
sum_ave = kriging.bd_Deriv_kriging_func(x, y, x_deriv, y_deriv, 'cst', 'cub', 100, 0.005)


