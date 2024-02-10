"""
Parametric kriging of surface
=============================

Test

"""

# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# parametric kriging of surface
from polytex.kriging.paraSurface import buildKriging, interp

if __name__ == '__main__':
    x = [[1, 1, 2],
         [np.sqrt(2) / 2, 1 / 2, 0],
         [0, 0, 1]]

    y = [[0, 0, 1],
         [np.sqrt(2) / 2, 1 / 2, 0],
         [1, 1, 0]]

    z = [[0, 0.5, 1],
         [0, 0.5, 1],
         [0, 0.5, 1]]

    # define two symbolic variables (parameters) s and t
    s = [0, 0.3, 1]  # rows
    t = [0, 0.5, 1]  # columns

    nugg = [0, 0.]

    xexpr = buildKriging(s, t, x, ['lin', 'const'], ['cub', 'lin'], nugg)
    yexpr = buildKriging(s, t, y, ['lin', 'const'], ['cub', 'lin'], nugg)
    zexpr = buildKriging(s, t, z, ['lin', 'const'], ['cub', 'lin'], nugg)

    # time the interpolation function
    import time

    start = time.time()

    for i in range(1):
        xinterp = interp(s, t, xexpr, split_complexity=0)
        yinterp = interp(s, t, yexpr, split_complexity=0)
        zinterp = interp(s, t, zexpr, split_complexity=0)

    print('time: ', time.time() - start)
