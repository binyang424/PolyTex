# ！/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Curve Kriging

Bin Yang  2021-9-2
"""
import numpy as np
import sympy as sym


######################################################
#               Drift and Covariance                 #
######################################################
def func_select(drift_name, cov_name):
    '''
    This is the function for definition of drift function and covariance function 
    in dictionary drif_funcs and cov_funcs.
    '''
    # Definitions of drift functions by dictionary
    import sympy as sym

    drift_funcs = {
        'const': lambda x, y, a: [a[0]],
        'lin': lambda x, y, a: [a[0], a[1] * x, a[2] * y],
        'quad': lambda x, y, a: [a[0], a[1] * x, a[2] * y, a[3] * (x ** 2.0), a[4] * (y ** 2.0), a[5] * x * y],
    }
    # Definitions of covariance functions by dictionary
    cov_funcs = {
        'lin': lambda h: h,
        'cub': lambda h: h ** 3.0,
        'log': lambda h: h ** 2.0 * sym.log(h) if h != 0 else 0,  # Natural logarithm, element-wise.
    }
    # Int number of 'a_len', which is based on the drift function, will be used in building of kriging matrix.
    a_len = len(drift_funcs[drift_name](0, 0, [1, 1, 1, 1, 1, 1]))
    return drift_funcs[drift_name], cov_funcs[cov_name], a_len


######################################################
#                  Curve Kriging                     #
######################################################
def curveKrig(dataset, name_drift, name_cov, nuggetEffect=0):
    '''
    Parameters
    ----------
    data: numpy array
        X-Y-Z.
    name_drift : String
        Name of drift.
    name_cov : String
        Name of covariance.
    nuggetEffect : Float

    Returns
    -------
    expr : Expression
        The kriging expression.
    '''

    # Read the drift and covariance type.
    func_drift, func_cov, len_a = func_select(name_drift, name_cov)
    len_b = dataset.shape[0]
    krig_len = len_a + len_b
    mat_krig = np.zeros([krig_len, krig_len])

    nugget = np.zeros([krig_len, krig_len])
    nugget[:len_b,:len_b] = np.identity(len_b)

    adef = [1, 1, 1, 1, 1, 1]
    a = func_drift(dataset[:, 0], dataset[:, 1], adef)

    # Assembling the kriging matrix
    for i in np.arange(len_b):
        for j in np.arange(len_b):
            if i <= j:
                # The Euclidean distance between two points in 2 dimensions.
                h = np.sqrt((dataset[:, 0][i] - dataset[:, 0][j]) ** 2 +
                            (dataset[:, 1][i] - dataset[:, 1][j]) ** 2)
                mat_krig[i, j] = func_cov(h)

    for i in np.arange(len_a):
        mat_krig[:len_b, len_b + i] = a[i]
    mat_krig = mat_krig + mat_krig.T - np.diag(mat_krig.diagonal()) + nugget * nuggetEffect

    # print('The value of determinant is {} '.format(np.linalg.det(mat_krig)))

    mat_krig_inv = np.linalg.inv(mat_krig) # inverse matrix
    #mat_krig_inv = np.linalg.pinv(mat_krig)  # generalized inverse

    u = np.zeros((krig_len, 1))
    u[:len_b, 0] = dataset[:, 2]  # function values ('z' in 3D space)

    # Solution
    vector_ba = mat_krig_inv.dot(u)

    x, y = sym.symbols('x y')

    # drift: func_drift(x,y, adef)
    a_drift = np.zeros(len_a)
    for i in np.arange(len_a):
        a_drift[i] = vector_ba[len_b + i]

    doc_krig = a_drift * func_drift(x, y, adef)

    for i in np.arange(len_b):
        bi_cov = ((func_cov((x - dataset[:, 0][i]) ** 2 + (y - dataset[:, 1][i]) ** 2)) ** (1 / 2)) * vector_ba[i, 0]
        doc_krig = np.append(doc_krig, bi_cov)  # store all the terms including drift and generilised covariance
    expr = doc_krig.sum()
    return mat_krig, mat_krig_inv, vector_ba, expr
