# ！/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

######################################################
#                      Data Normalization                                #
######################################################

def norm(data_krig, norm_type='axial'):
    '''
    This is the normalization function. After input the data of DSC test, this function
    will normalize temperature, degree of cure and rate of cure.

    Parameters
    ----------
    data_krig : numpy array
        Time-Temperature-Alpha-dadt
    norm_type : string, optional
        The type of normalization. The default is 'axial'. The other option is 'global' (TODO).
    '''

    data_shape = np.shape(data_krig)
    norm = np.zeros(data_shape)

    if norm_type == 'global':
        # TODO global normalization
        pass

    if norm_type == 'axial':
        # Axial normalization
        for i in np.arange(data_shape[1]):
            for j in np.arange(data_shape[0]):
                norm[:, i][j] = (data_krig[:, i][j] - np.min(data_krig[:, i])) / (
                            np.max(data_krig[:, i]) - np.min(data_krig[:, i]))
    return norm

# 将normalized result 转换回去



######################################################
#              Data Compression                                     #
######################################################
def data_compr(matXC, data_norm, max_err, skip_comp):
    '''
    Data compression by kriging using linear drift and linear covariance.

    Parameters
    ----------
    data_norm : numpy array
        Time-Temperature-Alpha-dadt
    max_err : float
        The criterion for data compression, which is the maximum local error.
    skip_comp : int
        skip (skip_comp-1) data point for data compression.
        skip_comp >=1.

    Returns
    -------
    data_norm_comp : TYPE
        Data points .
    extre : numpy array
        Index of data_norm_comp or extrema choosed according to kriging compression.
    '''
    from scipy.signal import argrelextrema 
    import numpy as np
    import matplotlib.pyplot as plt

    
    # for local maxima
    max_ind = argrelextrema(data_norm[:,1], np.greater_equal)
    # for local minima
    min_ind = argrelextrema(data_norm[:,1], np.less)
    extrema_ind = np.hstack((max_ind, min_ind))
    # include end points
    extrema_ind = np.insert(extrema_ind,0,[0],axis=1)
    extrema_ind = np.insert(extrema_ind,-1,[len(data_norm[:,1])-1],axis=1)
    # avoid repeat
    extrema_ind = np.unique(extrema_ind)
    extrema_ind.sort()   # sort by ascending
    
    merr_comp = 1e30
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    color = 'tab:red'

    ax1.set_ylabel('Degree of cure', color=color, fontsize=12)
    ax1.set_xlabel('Data points', fontsize=12)
    
    while merr_comp>max_err:
        
        xmat_krig, xmat_krig_inv, xvector_ba, xexpr = curve2D.curveKrig(matXC, 'lin', 'cub',nuggetEffect=1e-2)
        ymat_krig, ymat_krig_inv, yvector_ba, yexpr = curve2D.curveKrig(matYC, 'lin', 'cub',nuggetEffect=1e-2)

        x, y = sym.symbols('x y') 
        
        # keep 1 point in every skip_comp points in the compression loop
        if skip_comp!=0:
            data_norm_comp_ind = np.arange(0, data_norm.shape[0], skip_comp)  # local index
            data_norm_comp = data_norm[data_norm_comp_ind]
            ax1.plot(data_norm_comp[:,2], color=color)
        else:
            print('At least 1 point need to be select')
            return
                
        err_comp = np.zeros(data_norm_comp.shape)
        err_comp[:,:2] = data_norm_comp[:,:2]    
        
        for i in np.arange(data_norm_comp_ind.shape[0]):
            if data_norm_comp_ind[i] not in extrema_ind and data_norm_comp[i,2] !=0:
                err_comp[i,2] = (abs(expr.subs({x:err_comp[i,0],y:err_comp[i,1]})-
                                     data_norm_comp[i,2]))/np.max(data_norm_comp[:,2])
        merr_comp = max(err_comp[:,2])
        print(err_comp[:,2].sum(), merr_comp, extrema_ind.shape)
        
        extre = extrema_ind
        # for next loop
        max_ind = argrelextrema(err_comp[:,2], np.greater)[0]*skip_comp  # from local index to global index
        
        # # skip the extrema points when they have already safisfy the error limitation
        # max_ind = argrelextrema(err_comp[:,2], np.greater)[0]
        # for i in max_ind:
        #     if err_comp[i,2] < max_err*0.1:
        #         max_ind = np.delete(max_ind, [i], axis=0)
        # max_ind = max_ind*skip_comp    # from local index to global index
        
        extrema_ind = np.hstack((extrema_ind, max_ind))
        extrema_ind = np.unique(extrema_ind)
        extrema_ind.sort()   # sort by ascending
        data_norm_comp = data_norm[extre]
        
        ax1.plot(err_comp[:,2],)
    return data_norm_comp, extre


#%%
######################################################
#                  Kriging & Cross Validation                      #
######################################################
def fun_crva(data_norm, drift_para,cov_para):
    '''
    Parameters
    ----------
    data_norm : numpy array
        Time-Temperature-Alpha-dadt.
    drift_para : list
        List of string elements.
    cov_para : list
        List of string elements.

    Returns
    -------
    expr : Expression
        The kriging expression.
    '''
    
    import time as tm
    from sklearn.model_selection import LeaveOneOut
    
    fo = open('crossvalidation'+'.csv', "w")  # Log file for cross validation
    fo.write('Drift, Covariance, Cumulative error, local error,')
    fo.write('\n')
    
    fig = plt.figure()  # residual diagram
    ax1 = fig.add_subplot(2,1,1)  # Accumulated error
    ax2 = fig.add_subplot(2,1,2)  # Local error

    for drift in drift_para:
        for covariance in cov_para:
            start = tm.perf_counter() 
            print('-------------------------------------------------------------')
            print('Drift, Covariance, Cumulative error, Local error, percent')
            
            error_glo = []
            error_lo = []
            error = 0
            error_ith = 0 # error for i-th iteration
        
            loo = LeaveOneOut()
            
            for train, test in loo.split(data_norm):              
                mat_krig, mat_krig_inv, vector_ba, expr = surface_krig(data_norm[train], drift, covariance)
                
                x, y = sym.symbols('x y')
                error_ith = abs(expr.subs({x:data_norm[test,0],y:data_norm[test,1]})-data_norm[test,2])
                error += error_ith
                error_glo.append(error)
                error_lo.append(error_ith)
                
                print ('{}, {}, {}, {}, {}% '.format(drift, covariance, round(error,4), round(error_ith,4),
                                                      np.round(test/(len(data_norm[:,0]))*100)[0]))
                # if test%5 == 0:
                #     fo.write('%s, %s, %.4f, %.4f\n' % (drift, covariance, round(error,4), round(error_ith,4)))
            
            ax1.plot(np.arange(test+1), error_glo, label = str(drift+'+'+covariance))
            ax2.plot(np.arange(test+1), error_lo,  label = str(drift+'+'+covariance))
            ax1.set_ylabel('Cumulative error')
            ax2.set_ylabel('Local error')
            ax2.set_xlabel('Data points')
            ax1.legend()
            ax2.legend()
            
            fo.write('%s, %s, %.4f, %.4f\n' % (drift, covariance, round(error,4), round(error_ith, 4))) 
            print ('The Cumulative error: {0} for {1}th loop. The average error per loop: {2}'.format(
                round(error,4), test, round(error_ith, 4)))
            #-----------------------------------------
            end = tm.perf_counter()
            print('CPU time for cross validation: {}'.format(str(end-start)))
    fo.close()
    return expr
