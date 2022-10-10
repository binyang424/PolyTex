# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym


#  Drift and Covariance
def func_select(drift_name, cov_name):
    """
    This is the function for definition of drift function and covariance function
    in dictionary drif_funcs and cov_funcs.

    Parameters
    ----------
    drift_name: str.
        The name of the drift function. Possible values are: "const", "lin", "quad".
    cov_name: str.
        The name of the covariance function. Possible values are: "lin", "cub", "log".

    Returns
    -------
    drift_func:
        The drift function.
    cov_func:
        The covariance function.
    """
    # Definitions of drift functions by dictionary
    import sympy as sym

    drift_funcs = {
        'const': lambda x, a: [a[0]],
        'lin': lambda x, a: [a[0], a[1] * x],
        'quad': lambda x, a: [a[0], a[1] * x, a[2] * (x ** 2.0)]
    }
    # Definitions of covariance functions by dictionary
    cov_funcs = {
        'lin': lambda h: h,
        'cub': lambda h: h ** 3.0,
        # Natural logarithm, element-wise. Does not work in this version!!!
        'log': lambda h: h ** 2.0 * sym.log(h) if h != 0 else 0,
    }
    # Int number of 'a_len', which is based on the drift function, will be used in building of kriging matrix.
    a_len = len(drift_funcs[drift_name](0, [1, 1, 1, 1, 1, 1]))
    return drift_funcs[drift_name], cov_funcs[cov_name], a_len


def dist1D(x):
    """
    Calculate the distance between each pair of points.

    Parameters
    ----------
    x: numpy array.
        The coordinates in parametric space of the points. The shape is (m, 1).

    Returns
    -------
    numpy array.
        The distance between each pair of points. The shape is (m, m).
    """
    b_len = x.size

    xc = np.repeat(x, repeats=b_len, axis=0)
    xc = xc.reshape((b_len, b_len))
    xr = xc.T

    return np.abs(xc - xr)


def kVector(x, symVar, drift_name, cov_name):
    """
    Calculate the kriging matrix.

    Parameters
    ----------
    x: numpy array.
        The coordinates in parametric space of the points. The shape is (m, 1).
    symVar: String.
        The variable in parametric space.
    cov_name: String.
        The name of covariance function.

    Returns
    -------
    :return: numpy array.
        The kriging matrix. The shape is (m, 1).
    """
    s = sym.Symbol(symVar)
    drift_func, cov_func, a_len = func_select(drift_name, cov_name)
    # TODO: drift term
    kVec = cov_func(np.abs(x - s))

    adef = [1, 1, 1, 1, 1, 1, 1, 1]
    a = drift_func(s, adef)
    # append elements of drift function to kVector
    kVec = np.append(kVec, a)
    return kVec


def buildM(x, drift_name, cov_name):
    """
    Build the kriging matrix.

    Parameters
    ----------
    x: The coordinates of the points. The shape is (m, 2).
    drift_name: str. The name of the drift function.
        Possible values are: "const", "lin", "quad".
    cov_name: str. The name of the covariance function.
        Possible values are: "lin", "cub", "log".

    Returns
    -------
    :return drift_func: The drift function.
    :return cov_func: The covariance function.
    :return a_len: The length of the drift function.
    :return M: The matrix of the kriging system. The shape is (n,n).
    """
    # ------------drift and covariance function selection------------
    drift_func, cov_func, a_len = func_select(drift_name, cov_name)

    # -------- distance between each pair of points --------
    distance = dist1D(x)

    # ------------initialize the kriging matrix------------
    b_len = x.shape[0]
    n = b_len + a_len
    M = np.zeros((n, n))

    # -------- assembling the kriging matrix --------
    M[:b_len, :b_len] = cov_func(distance)

    # -------- elements depending on drift function --------
    adef = [1, 1, 1, 1, 1, 1, 1, 1]
    a = drift_func(x, adef)

    for i in np.arange(a_len):
        M[:b_len, b_len + i] = a[i]
        M[b_len + i, :b_len] = a[i]
    return M, drift_func, cov_func, a_len


def nugget(M, nugg, b_len):
    """
    Introduce the nugget effect to the kriging matrix.

    Parameters
    ----------
    M : numpy array.
        The kriging matrix.
    nugg: float.
        The nugget effect.

    Returns
    -------
    M : numpy array.
        The kriging matrix with nugget effect.
    """
    # -------- identity matrix with the same size as M --------
    I = np.identity(b_len)
    # -------- multiply nugg to the diagonal of I --------
    I = I * nugg
    # -------- add I to M --------
    M[:b_len, :b_len] = M[:b_len, :b_len] + I
    return M


def buildP(x, a_lenS, a_lenT):
    """
    Build the result vector of the kriging linear system.

    Parameters
    ----------
    z: numpy array.
        The values of the target function. The shape is (m,n).
    a_len: int.
        The length of the drift function.

    Returns
    -------
    P : numpy array.
        The result vector of the kriging linear system. The shape is (n,).
    """
    P = np.append(x, np.zeros((a_lenS, x.shape[1])), axis=0)
    P = np.append(P, np.zeros((P.shape[0], a_lenT)), axis=1)
    return P


def buildKriging(s, t, x, drift_names, cov_names, nugg=[0, 0]):
    """
    Build the kriging model and return the expression in string format.

    Parameters
    ----------
    s, t: numpy array.
        The parameters of the two profiles for surface parametric kriging.
    x: array like.
        The known values of the variables in parametric space.
    drift_names: list.
        The name of the drift functions for profile 1 and profile 2 in the
        following format: [drift_name1, drift_name2]. The possible values are: 'const', 'lin', 'cub'.
    cov_names: list.
        The name of the covariance functions in the following format:
        [covariance_name1, covariance_name2]. The possible values are: 'lin', 'cub', 'log'.
    nugg: list.
        The nugget effects (variance) for each profile contained in a list.

    Returns
    -------
    expr: The expression of kriging function in string format.
    """
    s, t, x = np.array(s), np.array(t), np.array(x)

    # ------- build the kriging matrix -------
    MS, drift_funcS, cov_funcS, a_lenS = buildM(s, drift_names[0], cov_names[0])
    MT, drift_funcT, cov_funcT, a_lenT = buildM(t, drift_names[1], cov_names[1])

    # ------- introduce nugget effect -------
    MS = nugget(MS, nugg[0], a_lenS)
    MT = nugget(MT, nugg[1], a_lenT)

    # ------- build parametric kriging vectors -------
    k1s = kVector(s, 's', drift_names[0], cov_names[0])
    k2t = kVector(t, 't', drift_names[1], cov_names[1])

    # ------- build the matrix of known variables P -------
    Px = buildP(x, a_lenS, a_lenT)

    # ------- build the parametric kriging model -------
    expr = np.linalg.multi_dot([k1s.reshape([1, -1]), np.linalg.inv(MS),
                                Px, np.linalg.inv(MT), k2t.reshape([-1, 1])]).sum()
    return expr


def interp(s, t, expr):
    """
    Interpolation (substitute the symbolic variables in the expression).

    Parameters
    ----------
    s, t: numpy array.
        The parameters of the two profiles for surface parametric kriging.
    expr: String.
        The expression of the target function.

    Returns
    -------
    xinterp: numpy array.
        The values of the kriging function. The shape is (s.size,t.size).
    """
    sVar, tVar = sym.symbols('s t')
    xinterp = np.zeros((len(s), len(t)))
    for i in range(len(s)):
        for j in range(len(t)):
            xinterp[i, j] = sym.simplify(expr.subs({sVar: s[i], tVar: t[j]}))
    return xinterp


if __name__ == '__main__':
    x = [[1, 1], [np.sqrt(2) / 2, 1 / 2], [0, 0]]
    y = [[0, 0], [np.sqrt(2) / 2, 1 / 2], [1, 1]]
    z = [[0, 1], [0, 1], [0, 1]]

    # define two symbolic variables (parameters) s and t
    s = [0, 0.5, 1]
    t = [0, 1]

    nugg = [0, 0]

    xexpr = buildKriging(s, t, x, ['lin', 'const'], ['cub', 'lin'], nugg)
    yexpr = buildKriging(s, t, y, ['lin', 'const'], ['cub', 'lin'], nugg)
    zexpr = buildKriging(s, t, z, ['lin', 'const'], ['cub', 'lin'], nugg)

    xinterp = interp(s, t, xexpr)
    yinterp = interp(s, t, yexpr)
    zinterp = interp(s, t, zexpr)
