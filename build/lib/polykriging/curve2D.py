# ！/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Curve Kriging

Bin Yang  2021-9-2
"""
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def addPoints(coordinate, threshold=0.03):
    '''
    coordinate: [normalized distance, X, Y, Z]
    threshold: the maximum delta normalized distance 
    '''
    deltaD = np.diff(coordinate[:, 0])

    for iDelta in np.arange(deltaD.size):
        if deltaD[iDelta] < threshold:
            try:
                temp = np.vstack((temp, coordinate[iDelta, :]))
            except NameError:
                temp = coordinate[iDelta, :]
        else:
            dtemp = np.linspace(coordinate[iDelta, 0], coordinate[iDelta + 1, 0], int(deltaD[iDelta] / 0.05) + 1,
                                endpoint=True)
            xtemp = np.interp(dtemp, coordinate[[iDelta, iDelta + 1], 0], coordinate[[iDelta, iDelta + 1], 1])
            ytemp = np.interp(dtemp, coordinate[[iDelta, iDelta + 1], 0], coordinate[[iDelta, iDelta + 1], 2])
            ztemp = np.empty(xtemp.shape)
            ztemp[:] = coordinate[0, -1]

            temp = np.vstack((temp, np.vstack((dtemp.T, xtemp.T, ytemp.T, ztemp.T)).T))
    coordinate = np.vstack((temp, coordinate[-1, :]))

    return coordinate


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
        # Natural logarithm, element-wise. Does not work in this version!!!
        'log': lambda h: h ** 2.0 * sym.log(h) if h != 0 else 0,
    }
    # Int number of 'a_len', which is based on the drift function, will be used in building of kriging matrix.
    a_len = len(drift_funcs[drift_name](0, [1, 1, 1, 1, 1, 1]))
    return drift_funcs[drift_name], cov_funcs[cov_name], a_len


def curve1Dsolve(dataset, krig_len, mat_krig, inverseType="inverse"):
    '''
    Solve the kriging equation: [Matrix_kriging] [b_a] = [u.. 0..]

    :param dataset: numpy array. The sample points. X-Y.
    :param krig_len: int. The length of the kriging vector.
    :param mat_krig: numpy array. The kriging matrix.
    :param inverseType: String. The type of the inverse matrix.
        "inverse" : inverse matrix
        "pseudoinverse" : generalized inverse/pseudoinverse
    :return b_a: numpy array. The kriging vector.
    :return mat_krig_inv: numpy array. The inverse matrix of the kriging matrix.
    '''
    # TODO: seperate the build of U vector and the solve of the kriging equation
    len_b = dataset.shape[0]
    u = np.zeros((krig_len, 1))
    u[:len_b, 0] = dataset[:, 1]

    # Solution
    if inverseType == "pseudoinverse":
        mat_krig_inv = np.linalg.pinv(mat_krig)  # generalized inverse/pseudoinverse
    else:
        mat_krig_inv = np.linalg.inv(mat_krig)  # inverse matrix

    vector_ba = mat_krig_inv.dot(u)

    return mat_krig_inv, vector_ba


def curve1Dexpression(len_b, func_drift, func_cov, adef, dataset, vector_ba):
    '''
    return the Kriging function expression.
    '''
    x = sym.symbols('x')

    # drift: func_drift(x, adef)
    a_drift = vector_ba[len_b:]
    drift = np.squeeze(a_drift).dot(func_drift(x, adef))

    doc_krig = []
    doc_krig = np.append(doc_krig, drift)

    for i in np.arange(len_b):
        fluctuation = (
                          func_cov(
                              sym.Abs(x - dataset[:, 0][i])
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
    dataset: numpy array
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
    krig_len = len_a + len_b  # len_a determined by the selected drift
    # empty Kriging matrix
    mat_krig = np.zeros([krig_len, krig_len])

    nugget = np.zeros([krig_len, krig_len])
    nugget[:len_b, :len_b] = np.identity(len_b)

    # define the last sevel columns of kriging matrix
    # depending on the type of drift function
    adef = [1, 1, 1, 1, 1, 1]
    a = func_drift(dataset[:, 0], adef)

    # Assembling the kriging matrix
    for i in np.arange(len_b):
        h = func_cov(abs(dataset[:, 0] - dataset[i, 0]))
        for j in np.arange(len_b):
            if i <= j:
                mat_krig[i, j] = h[j]

    for i in np.arange(len_a):
        mat_krig[:len_b, len_b + i] = a[i]

    mat_krig = mat_krig + mat_krig.T - np.diag(mat_krig.diagonal()) + nugget * nuggetEffect

    ##    print("the final mat_krig:\n", mat_krig)
    ##    print('The value of determinant is {} '.format(np.linalg.det(mat_krig)))

    # solve kriging linear equation system to get the vector_ba 
    mat_krig_inv, vector_ba = curve1Dsolve(dataset, krig_len, mat_krig, inverseType="inverse")
    # get the kriging function expression 
    expr = curve1Dexpression(len_b, func_drift, func_cov, adef, dataset, vector_ba)

    return mat_krig, mat_krig_inv, vector_ba, expr


def curve1Dinter(dataset, name_drift, name_cov, nuggetEffect=0, interp=' '):
    '''
    Parameters
    ----------
    dataset: numpy array
        X-Y.
    name_drift : String
        Name of drift.
    name_cov : String
        Name of covariance.
    nuggetEffect : Float
        smoothing strength control
    interp: Numpy array
        The points that need to be interpolated, 1D numpy array.
        If interp is not given, the x-coordinate of the sample points is used.
    Returns
    -------
    expr : Expression
        The kriging expression.
    '''

    mat_krig, mat_krig_inv, vector_ba, expr = \
        curveKrig1D(dataset, name_drift, name_cov, nuggetEffect)

    x = sym.symbols('x')

    if type(interp) is np.ndarray:
        yinter = np.empty(interp.shape[0])
        for pts in range(interp.shape[0]):
            yinter[pts] = expr.subs({x: interp[pts]})
    else:
        yinter = np.empty(dataset.shape[0])
        for pts in range(dataset.shape[0]):
            yinter[pts] = expr.subs({x: dataset[pts, 0]})

    yinter[np.abs(yinter) < 1e-15] = 0.

    return yinter, expr


# ---------------------Derivative Kriging--------------
# Initially programed by: Yixun Sun
# Modified by Bin Yang
# -----------------------------------------------------
def solveB(M, U):
    B = np.linalg.solve(M, U)
    print('solution Matrix b writes:')
    print(B)
    return B


def h(x1, x2):
    return np.abs(x1 - x2)


def buildKrigFunc_deriv(x, xKnown, xKnown_deriv, B, deriveFuncs, covFuncs, covFuncs_deriv):
    funcKrig = 0
    for i in range(len(xKnown)):
        funcKrig = covFuncs(h(x, xKnown[i])) * B[i] + funcKrig

    for i in range(len(xKnown), len(xKnown) + len(xKnown_deriv)):
        funcKrig = covFuncs_deriv(h(x, xKnown_deriv[i - len(xKnown)])) * B[i] * np.sign(
            xKnown_deriv[i - len(xKnown)] - x) + funcKrig

    for i in range(len(xKnown) + len(xKnown_deriv), len(B)):
        funcKrig = funcKrig + B[i] * deriveFuncs(x)[i - (len(xKnown) + len(xKnown_deriv))]
    return funcKrig


def buildM_deriv(x, x_deriv, name_drift, name_cov, covFuncs_deriv, covFuncs_deriv2, nugg):
    """
    Build the matrix M for the derivative kriging system

    Parameter
    ----------
    x: array
        x points
    xDeriv: array
        x points for derivative
    name_drift: function
        derivative functions
    name_cov: function
        covariance functions
    covFuncs_deriv: function
        derivative of covariance functions
    covFuncs_deriv2: function
        second derivative of covariance functions
    nugg: float
        nugget effect (variance)
    """

    # Initialization of matrix M
    xlen, xlen_deriv, driftLen = len(x), len(x_deriv), len(name_drift(x[0]))
    lenMatM = driftLen + xlen + xlen_deriv
    M = np.zeros((lenMatM, lenMatM))

    """
    Build the matrix M for the derivative kriging system
    """

    for i in range(xlen):
        for j in range(xlen):
            if i == j:
                M[i, j] = 0 + nugg
            else:
                M[i, j] = name_cov(h(x[i], x[j]))

    for i in range(xlen):
        for j in range(xlen_deriv):
            # TODO: replace the np.sign with np.sign()
            M[i, j + xlen] = covFuncs_deriv(x[i] - x_deriv[j]) * np.sign(-x[i] + x_deriv[j])
            M[j + xlen, i] = covFuncs_deriv(x[i] - x_deriv[j]) * np.sign(-x[i] + x_deriv[j])

    for i in range(xlen_deriv):
        for j in range(xlen_deriv):
            # TODO: replace the np.sign with np.sign()
            M[i + xlen, j + xlen] = covFuncs_deriv2(x_deriv[i] - x_deriv[j]) * np.sign(-x_deriv[i] + x_deriv[j])
            M[j + xlen, i + xlen] = covFuncs_deriv2(x_deriv[i] - x_deriv[j]) * np.sign(-x_deriv[i] + x_deriv[j])

    for i in range(xlen):
        for j in range(driftLen):
            M[i, j + xlen + xlen_deriv] = name_drift(x[i])[j]
            M[j + xlen + xlen_deriv, i] = name_drift(x[i])[j]

    print('Matrix M writes:')
    print(M)
    print('Solve M*b=u to obtain Kriging function')

    return M


def buildU_deriv(y, y_deriv, deriveFuncs):
    ylen, ylen_deriv, deriveFuncLen = len(y), len(y_deriv), len(deriveFuncs(y[0]))
    lenMatU = deriveFuncLen + ylen + ylen_deriv
    U = np.zeros(lenMatU)
    U[:ylen] = y
    U[ylen:ylen + ylen_deriv] = y_deriv
    print('Matrix u writes:')
    print(U)
    return U

def bd_Deriv_kriging_func(x, y, xDeriv, yDeriv, choixDerive, choixCov, plot_x_pts, nugg):
    """
    Derivative kriging function.
    :param x: array, x points
    :param y: array, y points
    :param xDeriv: array, x points for derivative
    :param yDeriv: array, the derivative of xDeriv points
    :param choixDerive: string, 'cst', 'lin' or 'quad'
    :param choixCov: string, 'lin' or 'cub'
    :param plot_x_pts: array, number of points for plot
    :param nugg: float, nugget effect (variance)
    :return kringFunctionStr: string, string of the kriging function
    :return x_var_sym: string, string of the x variable
    """

    # plot the original dataset using scatter
    plt.scatter(x, y, color='k', marker='x', alpha=0.7, s=12, label='')

    # ---------------Choice of drift -----------------------
    deriveFuncs = {'cst': lambda x: [1], 'lin': lambda x: [1, x]}
    # -------------------Choice of covariance---------------
    covFuncs = {'cub': lambda x: x ** 3., 'lin': lambda x: x}

    # ------------------Derivative of covariance-------------
    covFuncs_deriv = {'cub': lambda x: 3 * x ** 2., 'lin': lambda x: x ** 0}
    covFuncs_deriv2 = {'cub': lambda x: 6 * x ** 1., 'lin': lambda x: x * 0}

    # ---------------------Build matrix M---------------------
    M = buildM_deriv(x, xDeriv, deriveFuncs[choixDerive], covFuncs[choixCov], covFuncs_deriv[choixCov],
                     covFuncs_deriv2[choixCov], nugg)

    # ---------------------Build matrix u--------------
    U1 = buildU_deriv(y, yDeriv, deriveFuncs[choixDerive])

    # ----------------------solve b--------------------
    B1 = solveB(M, U1)

    # ----------------build string function------------
    lowerX, upperX = min(x), max(x)
    intervalX = (upperX - lowerX) / plot_x_pts
    x_krig = [i * intervalX for i in range(int(lowerX / intervalX), int((upperX) / intervalX) + 1)]
    y_krig = [buildKrigFunc_deriv(x_krig[i], x, xDeriv, B1, deriveFuncs[choixDerive], covFuncs[choixCov],
                                  covFuncs_deriv[choixCov]) for i in range(len(x_krig))]
    sum_ave = 0
    for i in range(1, len(x_krig)):
        hh = x_krig[i] - x_krig[i - 1]
        a_b = y_krig[i] + y_krig[i - 1]
        sum_ave = sum_ave + 0.5 * hh * a_b

    sum_ave = sum_ave / h(min(x), max(x))
    plt.plot(x_krig, y_krig, linestyle='--', lw=1, label='Nugget effect = ' + str(nugg))

    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc='upper left', ncol=1)

    plt.show()

    return sum_ave


if __name__ == "__main__":
    dataset = np.array([[0, 0],
                        [0.25, -1],
                        [0.75, -1],
                        [1, 0]])

    name_drift = 'const'
    name_cov = 'lin'
    nuggetEffect = 0

    mat_krig, mat_krig_inv, vector_ba, expr = \
        curveKrig1D(dataset, name_drift, name_cov, nuggetEffect=0.0)

    yinter = curve1Dinter(dataset, name_drift, name_cov, nuggetEffect=0, interp=' ')

    print("the kriging matrix:\n", mat_krig)
    print("the kriged y values:\n", yinter)
    print("vector_ba:\n", vector_ba)

'''
the final mat_krig for checking purpose:
 [[0.       0.015625 0.421875 1.       1.      ]
 [0.015625 0.       0.125    0.421875 1.      ]
 [0.421875 0.125    0.       0.015625 1.      ]
 [1.       0.421875 0.015625 0.       1.      ]
 [1.       1.       1.       1.       0.      ]]
 
[[0.   0.25 0.75 1.   1.  ]
 [0.25 0.   0.5  0.75 1.  ]
 [0.75 0.5  0.   0.25 1.  ]
 [1.   0.75 0.25 0.   1.  ]
 [1.   1.   1.   1.   0.  ]]
 '''
