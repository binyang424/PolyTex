"""
Derivative_Kriging
==================

Test

"""

import matplotlib.pyplot as plt
from polykriging import curve2D

# ----------------- Example 1 --------------------
x = [0, 2.5, 4, 4.5, 5.08]
y = [5, 8, 9, 5, -3]

x_deriv = [0]
y_deriv = [-1]

# bd_Deriv_kriging_func(x, y, xDeriv, yDeriv, choixDerive, choixCov, plot_x_pts, nugg)
sum_ave = curve2D.bd_Deriv_kriging_func(x, y, x_deriv, y_deriv, 'cst', 'cub', 100, 0)

print(sum_ave)

plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='upper left', ncol=1)


# clear matplotlib figure
plt.clf()

# ----------------- Example 2 --------------------
x = [0, 1]
y = [0, 0]

x_deriv = [0.5]
y_deriv = [1]

# bd_Deriv_kriging_func(x, y, xDeriv, yDeriv, choixDerive, choixCov, plot_x_pts, nugg)
sum_ave = curve2D.bd_Deriv_kriging_func(x, y, x_deriv, y_deriv, 'cst', 'cub', 100, 0.005)


