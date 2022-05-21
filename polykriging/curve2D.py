# ！/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Curve Kriging

Bin Yang  2021-9-2
"""
import numpy as np
import sympy as sym


#  Drift and Covariance
def func_select(drift_name, cov_name):
    '''
    This is the function for definition of drift function and covariance function 
    in dictionary drif_funcs and cov_funcs.
    '''
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
        # Natural logarithm, element-wise. Do not work in this version!!!
        'log': lambda h: h ** 2.0 * sym.log(h) if h != 0 else 0,  
    }
    # Int number of 'a_len', which is based on the drift function, will be used in building of kriging matrix.
    a_len = len(drift_funcs[drift_name](0, [1, 1, 1, 1, 1, 1]))
    return drift_funcs[drift_name], cov_funcs[cov_name], a_len


def curve1Dsolve(dataset, krig_len, mat_krig, inverseType = "inverse"):
    '''
    Solve [Matrix_kriging] [b_a] = [u.. 0..] 
    '''
    len_b = dataset.shape[0] 
    u = np.zeros((krig_len, 1))
    u[:len_b, 0] = dataset[:, 1]

    # Solution
    if inverseType == "pseudoinverse":
        mat_krig_inv = np.linalg.pinv(mat_krig)  # generalized inverse/pseudoinverse
    else:
        mat_krig_inv = np.linalg.inv(mat_krig) # inverse matrix

    vector_ba = mat_krig_inv.dot(u)

    return mat_krig_inv, vector_ba


def curve1Dexpression(len_b, func_drift, func_cov, adef, dataset, vector_ba):
    '''
    return the Kriging function expression.
    '''
    x = sym.symbols('x')

    # drift: func_drift(x, adef)
    a_drift = vector_ba[len_b:]
    drift = a_drift * func_drift(x, adef)

    doc_krig = []
    doc_krig = np.append(doc_krig, drift)

    for i in np.arange(len_b):
        fluctuation = (
                func_cov(
                    sym.Abs( x - dataset[:, 0][i] ) 
                                )
            ) * vector_ba[i, 0]

        # store all the terms including drift and generilised covariance
        doc_krig = np.append(doc_krig, fluctuation)
    expr = doc_krig.sum()

    return expr


# Curve Kriging
def curveKrig1D(dataset, name_drift, name_cov, nuggetEffect=0):
    '''
    Parameters
    ----------
    data: numpy array
        X-Y.
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
    
    len_b = dataset.shape[0]  # same as the number of sample points
    krig_len = len_a + len_b    # len_a determined by the selected drift 
    # empty Kriging matrix
    mat_krig = np.zeros([krig_len, krig_len])
    
    nugget = np.zeros([krig_len, krig_len])
    nugget[:len_b,:len_b] = np.identity(len_b)

    # define the last sevel columns of kriging matrix
    # depending on the type of drift function
    adef = [1, 1, 1, 1, 1, 1]
    a = func_drift(dataset[:, 0], adef)

    # Assembling the kriging matrix
    for i in np.arange(len_b):
        h = func_cov( abs( dataset[:, 0] - dataset[i, 0] ) )
        for j in np.arange(len_b):
            if i <= j:
                mat_krig[i, j] = h[j]

    for i in np.arange(len_a):
        mat_krig[:len_b, len_b + i] = a[i]

##    print("the half mat_krig:\n", mat_krig)
##    print('The value of the half determinant is {} '.format(np.linalg.det(mat_krig)))

    mat_krig = mat_krig + mat_krig.T - np.diag(mat_krig.diagonal()) + nugget * nuggetEffect

##    print("the final mat_krig:\n", mat_krig)
##    print('The value of determinant is {} '.format(np.linalg.det(mat_krig)))

    # solve kriging linear equation system to get the vector_ba 
    mat_krig_inv, vector_ba = curve1Dsolve(dataset, krig_len, mat_krig, inverseType = "inverse")
    # get the kriging function expression 
    expr = curve1Dexpression(len_b, func_drift, func_cov, adef, dataset, vector_ba)

    return mat_krig, mat_krig_inv, vector_ba, expr


def curve1Dinter(dataset, name_drift, name_cov, nuggetEffect=0, xinter = ' '):
    # xinter: The points that need to be interpolated, 1D numpy array
    # if xinter is not given, the x-coordinate of the sample points is used.
    mat_krig, mat_krig_inv, vector_ba, expr =  \
                curveKrig1D(dataset, "const", "cub", nuggetEffect)
    
    x = sym.symbols('x')
    yinter = np.empty(dataset.shape[0])

    if type(xinter) is np.ndarray:
        for pts in range(xinter.shape[0]):
            yinter[pts] = expr.subs({x:x[pts, 0]})
    else:
        for pts in range(dataset.shape[0]):
            yinter[pts] = expr.subs({x:dataset[pts, 0]})

    yinter[yinter<1e-15] = 0.
    
    return yinter


if __name__ == "__main__":
    dataset = np.array([[0, 0],
                                     [0.25, 1],
                                     [0.75, 1],
                                     [1, 0] ])
    
    mat_krig, mat_krig_inv, vector_ba, expr =  \
                    curveKrig1D(dataset, "const", "cub", nuggetEffect=0.0)
    
    yinter = curve1Dinter(dataset, "const", "cub", nuggetEffect=0, xinter = ' ')
    
'''
the final mat_krig for checking purpose:
 [[0.       0.015625 0.421875 1.       1.      ]
 [0.015625 0.       0.125    0.421875 1.      ]
 [0.421875 0.125    0.       0.015625 1.      ]
 [1.       0.421875 0.015625 0.       1.      ]
 [1.       1.       1.       1.       0.      ]]
 '''
