# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym

# parametric kriging of surface
from polykriging.paraSurface import buildKriging, interp


if __name__ == '__main__':
    x = [[1, 1], [np.sqrt(2) / 2, 1 / 2], [0, 0]]
    y = [[0, 0], [np.sqrt(2) / 2, 1 / 2], [1, 1]]
    z = [[0, 1], [0, 1], [0, 1]]

    # define two symbolic variables (parameters) s and t
    s = [0, 0.5, 1]
    t = [0, 1]

    nugg = [0, 0.]

    xexpr = buildKriging(s, t, x, ['lin', 'const'], ['cub', 'lin'], nugg)
    yexpr = buildKriging(s, t, y, ['lin', 'const'], ['cub', 'lin'], nugg)
    zexpr = buildKriging(s, t, z, ['lin', 'const'], ['cub', 'lin'], nugg)

    xinterp = interp(s, t, xexpr)
    yinterp = interp(s, t, yexpr)
    zinterp = interp(s, t, zexpr)
