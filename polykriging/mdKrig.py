# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym


def func_select(drift_name, cov_name):
    """
    Function for definition of drift and covariance function
    in dictionary drif_funcs and cov_funcs.
    :param drift_name: str. The name of the drift function.
        Possible values are: "const", "lin", "quad".
    :param cov_name: str. The name of the covariance function.
        Possible values are: "lin", "cub", "log".
    :return: drift_func, cov_func, a_len.
        drift_func: Function. The drift function.
        cov_func: Function. The covariance function.
        a_len: int. The length of the drift function.
    """
    # Definitions of drift functions by dictionary
    drift_funcs = {
        'const': lambda x, y, a: [a[0]],
        'lin': lambda x, y, a: [a[0], a[1] * x, a[2] * y],
        'quad': lambda x, y, a: [a[0], a[1] * x, a[2] * y,
                                 a[3] * (x ** 2.0), a[4] * (y ** 2.0), a[5] * x * y],
    }
    # Definitions of covariance functions by dictionary
    cov_funcs = {
        'lin': lambda h: h,
        'cub': lambda h: h ** 3.0,
        # Natural logarithm, element-wise.
        'log': lambda h: h ** 2.0 * sym.log(h) if h != 0 else 0,
    }
    # Int number of 'a_len', which is based on the drift function,
    # will be used in building of kriging matrix.
    a_len = len(drift_funcs[drift_name](0, 0, [1, 1, 1, 1, 1, 1]))
    return drift_funcs[drift_name], cov_funcs[cov_name], a_len


def dist(xy, type="Euclidean"):
    """
    Calculate the distance between each pair of points.
    :param xy: numpy array. The coordinates of the points. The shape is (m, 2).
    :param type: str. The type of the distance. The default is "Euclidean".
        Other possible values are:
            "1-norm" : The 1-norm distance.
            "inf-norm" : The infinity-norm distance.
    :return: numpy array. The distance between each pair of points. The shape is (m, m).
    """
    x, y = xy[:, 0], xy[:, 1]

    b_len = x.size

    xc = np.repeat(x, repeats=b_len, axis=0)
    xc = xc.reshape((b_len, b_len))
    xr = xc.T

    yc = np.repeat(y, repeats=b_len, axis=0)
    yc = yc.reshape((b_len, b_len))
    yr = yc.T

    if type == "Euclidean":
        distance = np.sqrt((xr - xc) ** 2 + (yr - yc) ** 2)
    elif type == "1-norm":
        distance = np.abs(xr - xc) + np.abs(yr - yc)
    elif type == "inf-norm":
        distance = np.max(np.abs(xr - xc), np.abs(yr - yc))

    return distance


def buildM(xy, drift_name, cov_name):
    """
    Build the kriging matrix.
    :param xy: The coordinates of the points. The shape is (m, 2).
    :param func_drift: The drift function.
    :param a_len: The length of the drift function.
    :return: The matrix of the kriging system. The shape is (n,n).
    """
    # ------------drift and covariance function selection------------
    drift_func, cov_func, a_len = func_select(drift_name, cov_name)

    # -------- distance between each pair of points --------
    distance = dist(xy)

    # ------------initialize the kriging matrix------------
    b_len = xy.shape[0]
    n = b_len + a_len
    M = np.zeros((n, n))

    # -------- assembling the kriging matrix --------
    M[:b_len, :b_len] = cov_func(distance)

    # -------- elements depending on drift function --------
    adef = [1, 1, 1, 1, 1, 1, 1, 1]
    a = drift_func(xy[:, 0], xy[:, 1], adef)

    for i in np.arange(a_len):
        M[:b_len, b_len + i] = a[i]
        M[b_len + i, :b_len] = a[i]

    return M, drift_func, cov_func, a_len


# TODO: verify the function
def nugget(M, nugg, b_len):
    """
    Introduce the nugget effect to the kriging matrix.
    :param M:   The kriging matrix.
    :param nugg:    The nugget effect.
    :return:    The kriging matrix with nugget effect.
    """
    # -------- identity matrix with the same size as M --------
    I = np.identity(b_len)
    # -------- multiply nugg to the diagonal of I --------
    I = I * nugg
    # -------- add I to M --------
    M[:b_len, :b_len] = M[:b_len, :b_len] + I
    return M


def buildU(z, a_len):
    """
    Build the result vector of the kriging linear system.
    :param z: The values of the target function. The shape is (m,).
    :param a_len: The length of the drift function.
    :return: The result vector of the kriging linear system. The shape is (n,).
    """
    n = z.size + a_len
    U = np.zeros((n, 1))
    U[:z.size, 0] = z
    return U


def solveB(M, U):
    """
    Solve the kriging linear system.
    :param M:  numpy array. The kriging matrix.
    :param U:  numpy array. The result vector of the kriging linear system.
    :return B: numpy array. The solution of the kriging linear system (vector contains b_i and a_i).
    """
    b = np.linalg.solve(M, U)
    print('solution Matrix b writes:')
    print(b)
    return b


def buildKriging(xy, z, drift_name, cov_name, nugg=0):
    """
    Build the kriging model and return the expression in string format.
    :param xy: array like. The coordinates of the points. The shape is (m, 2).
    :param z: array like. The values of the target function. The shape is (m,).
    :param drift_name: str. The name of the drift function.
            The possible values are: 'const', 'lin', 'cub'.
    :param cov_name: str. The name of the covariance function.
            The possible values are: 'lin', 'cub', 'log'.
    :param nugg: float. The nugget effect (variance).
    :return: The expression of kriging function in string format.
    """
    global M, drift_func, cov_func, a_len,U, B, doc_krig

    # ------- build the kriging matrix -------
    M, drift_func, cov_func, a_len = buildM(xy, drift_name, cov_name)

    # ------- introduce nugget effect -------
    b_len = xy.shape[0]
    M = nugget(M, nugg, b_len)

    # ------- build the result vector -------
    U = buildU(z, a_len)
    # ------- solve the kriging linear system -------
    B = solveB(M, U)

    # ------- build the kriging model -------
    x, y = sym.symbols('x y')
    doc_krig = drift_func(x,y,B[b_len:])

    for i in np.arange(xy.shape[0]):
        bi_cov = ((cov_func((x - xy[:, 0][i]) ** 2 + (y - xy[:, 1][i]) ** 2)) ** (1 / 2)) * B[i, 0]
        # store all the terms including drift and generalized covariance
        doc_krig = np.append(doc_krig, bi_cov)
    return doc_krig.sum()


def interp(xy, expr):
    """
    :param xy: numpy array. The coordinates of the points. The shape is (m, 2).
    :param expr: String. The expression of the target function.
    :return: The values of the kriging function. The shape is (m,).
    """
    x, y = sym.symbols('x y')

    yinter = np.empty(xy.shape[0])
    for pts in range(xy.shape[0]):
        yinter[pts] = expr.subs({x: xy[pts, 0], y: xy[pts, 1]})

    yinter[np.abs(yinter) < 1e-15] = 0.

    return yinter


if __name__ == '__main__':
    dataset = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    dataset = np.array(dataset)
    xy = dataset[:, :2]
    z = dataset[:, 2]

    expr = buildKriging(xy, z, 'lin', 'cub', nugg=0.01)

    zInterp = interp(dataset[:, :2], expr)

