#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kriging_tool

@author: Wei

Krig the input data with curve kriging or surface kriging.

"""

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from pathlib import Path

import json

import copy

class kriging_class:
    def __init__(self, controller):
        self.js_tow_data = None
        self.krig_data = {}
        self.krig_data["Parameterization"] = {}
        self.func = None
        self.a_len = {}
        self.drift_funcs = {}
        self.cov_funcs = {}
        self.para_krig_num = 30 # The number of points for parametric kriging


    def krigsep_data(self, controller):
        self.func = controller.func

    def read_data(self, controller):
        if controller.js_path != Path("."):
            with controller.js_path.open("r") as f:
                self.js_tow_data = json.load(f)
            self.krig_data["Info"] = copy.deepcopy(self.js_tow_data["Info"])


    def parameterization(self, process_data):
        # This function should be used before np.mat?
        '''
        This is the parameterization function. After input the data of contour, this function
        will make a parameterization.
        's' indicate the section. (xy,z=0)
        't' indicate the arc direction. (z,which is also the slices stack direction)
        Args:
            contour_data: The information of contour, type 'np.darray'.
        '''
        s_len = process_data.shape[0]
        t_len = 1
        # # Discrete parameterization
        # s_cor = np.linspace(0,1.0,s_len)
        # Paremeterizaiton based on distance
        dis_list = []
        for i in range(s_len-1):
            dis = np.sqrt((process_data[i+1,1]-process_data[i,1])**2 + (process_data[i+1,0]-process_data[i,0])**2)
            dis_list.append(dis)
        dis_list = np.array(dis_list)
        s_cor = np.zeros(s_len)
        for i in range(s_len-1):
            s_cor[i+1] = (s_cor[i] + dis_list[i])
        s_cor = s_cor/dis_list.sum()
        # print(dis_list)
        # print(s_cor)
        # print("..............")
        # To construct the parameterization coordinates of all the nodes, which the
        # position of the nodes is related to the coordinates in xyz system.
        data_para = np.zeros([t_len, s_len, 2])
        for i in range(s_len):
            data_para[:,i,0] = s_cor[i]
        # This condition works when process the surface kriging data.
        if t_len > 1:
            t_cor = np.linspace(0, 1.0, t_len)
            for i in range(t_len):
                data_para[i,:,1] = t_cor[i]
            # data_para[:,:,1] = np.linspace(0, 1.0, t_len)
        else:
            t_cor = 0
            data_para[0,:,1] = t_cor
        # Parameterization by arc length?
        # Angular parameterization?
        return data_para

    def parametric_krig(self, process_data, name_drift, name_cov, krig_num):
        """
        This is the parametric kriging part of the surface kriging module. This function
        is programmed to process the contour data to get the same number in one tow of
        each section.
        Args:
            process_data: Type 'DataFrame', contains the information of coordinates: 'X','Y','Z'.
                        Later, this DataFrame should contain other information of tow
            name_drift: Name of drift function, such as 'const', 'lin', 'quad', 'trig'.
            name_cov: Name of covariance function, such as 'lin', 'cub', 'log', 'sin'.
        """
        # if process_data.iloc[-1]['X'] != process_data.iloc[0]['X'] or process_data.iloc[-1]['Y'] != process_data.iloc[0]['Y']:
        #     process_data = process_data.append(process_data.iloc[0], ignore_index = True)
        # process_data['X'].append(process_data['X'][0])
        # process_data['X'].append(process_data['Y'][0])
        para_data = self.parameterization(process_data)
        func_drift, func_cov, len_a = self.func_select(name_drift, name_cov)
        len_b = process_data.shape[0]
        krig_len = len_a + len_b
        mat_krig = np.zeros([krig_len, krig_len])
        adef=[1,1,1,1]
        krig_a = func_drift(para_data[0,:,0],adef)  # Here should be para_data[x,:,0] x for slice?????
        for i in range(len_b):
            krig_b = func_cov(abs(para_data[0,:,0]-para_data[0,i,0])) # Here should be para_data[x,:,0] x for slice?????
            for j in range(len_b):
                mat_krig[i,j] = krig_b[j]
            for j in range(len_a):
                # if a[0] could be changed to a[0][len(x)], then this condition can
                # be removed.
                if j == 0:
                    mat_krig[i,j+len_b] = krig_a[j]
                    mat_krig[j+len_b,i] = krig_a[j]
                else:
                    mat_krig[i,j+len_b] = krig_a[j][i]
                    mat_krig[j+len_b,i] = krig_a[j][i]


        mat_krig_inv = np.linalg.inv(mat_krig)

        # Vector [x1, y1; x2, y2; .......; xN, yN; 0, 0; 0, 0; .....]
        Q_para = np.zeros([3, krig_len ,1]) #Q_para[0,:,0] -- X, Q_para[1,:,0] -- Y, Q_para[2,:,0] -- Z
        Q_para[0, :len_b, 0] = process_data[:,0]
        Q_para[1, :len_b, 0] = process_data[:,1]
        Q_para[2, :len_b, 0] = process_data[:,2]

        # Calculate the vector of [ai,bi...]
        vector_ba = np.zeros([3,krig_len,1])
        for i in range(3):
            vector_ba[i] = mat_krig_inv.dot(Q_para[i])

        # Interpolation
        inter_data = np.zeros([1, krig_num, 2])
        # Parameterization, seems needs to be optimized
        for i in range(krig_num):
            inter_data[0, i, 0] = i/(krig_num-1.0)
        # print(inter_data[0, :, 0])
        inter_a = func_drift(inter_data[0,:,0],adef)
        # Why are the shapes of these matrix is [1,n,2 or 3] while the similar
        # ones above is [2 or 3, n, 1]
        # TODO, here the shape can be reduced to [krig_num,3], even to [krig_num,2]
        # inter_mat = np.zeros([1,krig_num,3], dtype=np.int16)
        inter_mat = np.zeros([1,krig_num,3])
        for i in range(krig_num):
            inter_b = func_cov(abs(para_data[0,:,0]-inter_data[0,i,0])) # Here should be para_data[x,:,0] x for slice?????
            # a0, a1, a2, a3...
            inter_a_value = np.zeros([len(inter_a),1])
            inter_a_value[0,0] = 1
            for j in range(1,len(inter_a)):
                inter_a_value[j,0] = inter_a[j][i]
            inter_b = inter_b.reshape(len(inter_b),1)
            inter_vector = np.vstack((inter_b,inter_a_value))
            for n in range(3):
                inter_mat[0,i,n] = inter_vector.T.dot(vector_ba[n])

        return inter_mat
        # return inter_mat, vector_ba[1][:len_b], vector_ba[1][len_b:]

    def one_dimensional_krig(self, process_data, name_drift, name_cov):
        return vector_ba[:len_b], vector_ba[len_b:]

    def surf_parameterization(self, process_data):
    # def surf_parameterization(self, process_data, t_para_list):
        # Here the process_data is the variable in Module "krigsep_window_set" -- cor_data_mat,
        # cor_data_mat = np.zeros(cor_data_s_len, 2, cor_data_t_len), in which,
        # cor_data_s_len = self.num_PK, cor_data_t_len = len(self.js_tow_data["Info"][view_name + " Selected Slice"]).
        s_len = process_data.shape[0]
        # s_len = process_data.shape[1]
        t_len = process_data.shape[2]
        # print("t_len = {:g}".format(t_len))
        s_cor = np.linspace(0,1.0,s_len)
        t_cor = np.linspace(0,1.0,t_len)
        data_para = np.zeros([t_len, s_len, 2])

        for i in range(s_len):
            data_para[:, i, 0] = s_cor[i]
        for i in range(t_len):
            data_para[i, :, 1] = t_cor[i]
            # This condition works when process the surface kriging data.
        # if t_len > 1:
#             t_cor = t_para_list
#             for i in range(t_len):
#                 data_para[i, :,1] = t_cor[i]

        else:
            t_cor = 0
            data_para[0, :, 1] = t_cor
            # Parameterization by arc length?
            # Angular parameterization?
        # print(data_para)
        return data_para

    def surface_krig(self, process_data, s_name_drift, s_name_cov, t_name_drift, t_name_cov, s_inter, t_inter):
    # def surface_krig(self, process_data, s_name_drift, s_name_cov, t_name_drift, t_name_cov, s_inter, t_inter, t_para_list):
        para_contour = self.surf_parameterization(process_data)
        # para_contour = self.surf_parameterization(process_data, t_para_list)
        # print(para_contour)
        s_func_drift, s_func_cov, s_len_a = self.func_select(s_name_drift, s_name_cov)
        t_func_drift, t_func_cov, t_len_a = self.func_select(t_name_drift, t_name_cov)

        s_len_b = process_data.shape[0]
        t_len_b = process_data.shape[2]
        s_krig_len = s_len_a + s_len_b
        t_krig_len = t_len_a + t_len_b

        s_mat_krig = np.zeros([s_krig_len, s_krig_len])
        t_mat_krig = np.zeros([t_krig_len, t_krig_len])

        # For 's' in the section, only interpolate 'x'.
        adef = [1, 1, 1, 1]
        # The elements which is related to a in the kriging matrix.
        #   x_a[0] == 1, related to a_0.
        #   x_a[i] == func_drift, related to a_i.
        # For example, if name_drift = 'lin',then x_a is a list with 2 row.
        #   x_a[0] = 1, x_a[1] = array([ 0 , ..(x).., 1])
        s_a = s_func_drift(para_contour[0, :, 0], adef)
        for i in range(s_len_b):
            s_b = s_func_cov(abs(para_contour[0, :, 0] - para_contour[0, i, 0]))
            for j in range(s_len_b):
                s_mat_krig[i, j] = s_b[j]
            for j in range(s_len_a):
                if j == 0:
                    s_mat_krig[i, j + s_len_b] = s_a[j]
                    s_mat_krig[j + s_len_b, i] = s_a[j]
                else:
                    s_mat_krig[i, j + s_len_b] = s_a[j][i]
                    s_mat_krig[j + s_len_b, i] = s_a[j][i]

        t_a = t_func_drift(para_contour[:,0,1],adef)
        for i in range(t_len_b):
            t_b = t_func_cov(abs(para_contour[:,0,1]-para_contour[i,0,1]))
            for j in range(t_len_b):
                t_mat_krig[i,j] = t_b[j]
            for j in range(t_len_a):
                if j == 0:
                    t_mat_krig[i,j+t_len_b] = t_a[j]
                    t_mat_krig[j+t_len_b,i] = t_a[j]
                else:
                    t_mat_krig[i,j+t_len_b] = t_a[j][i]
                    t_mat_krig[j+t_len_b,i] = t_a[j][i]

        s_mat_krig_inv = np.linalg.inv(s_mat_krig)
        t_mat_krig_inv = np.linalg.inv(t_mat_krig)
        # The matrix [Q].
        Q_surf = np.zeros([3, s_krig_len, t_krig_len])
        for i in range(s_len_b):
            for j in range(t_len_b):
                for n in range(3):
                    Q_surf[n, i, j] = process_data[i,n,j]
        # surf_mat denote [S]-1*Q_surf*[T]-1
        surf_mat = np.zeros([3,s_krig_len,t_krig_len])
        for i in range(3):
            surf_mat[i] = s_mat_krig_inv.dot(Q_surf[i]).dot(t_mat_krig_inv)
        # Now the parametric kriging in one section is finished. Next step is to expand it to surface kriging.
        # **** What the difference between: 1. put both nodes and seeds in "part";
        #                                  2. only put nodes in "part", and put seeds in "mesh".

        # surface_plot(data_contour)
        s_func_drift, s_func_cov, s_len_a = self.func_select(s_name_drift, s_name_cov)
        t_func_drift, t_func_cov, t_len_a = self.func_select(t_name_drift, t_name_cov)
        inter_contour = np.zeros([t_inter, s_inter, 2])
        for i in range(t_inter):
            for j in range(s_inter):
                inter_contour[:, j, 0] = j / (s_inter - 1.0)
                inter_contour[i, :, 1] = i / (t_inter - 1.0)
        adef = [1, 1, 1, 1]
        s_a = s_func_drift(inter_contour[0, :, 0], adef)
        t_a = t_func_drift(inter_contour[:, 0, 1], adef)
        # The original code is np.zeros([t_inter, s_inter, 6]), becuase there is some additional information.
        # inter_mat = np.zeros([t_inter, s_inter, 6])
        inter_mat = np.zeros([t_inter, s_inter, 3])


        for i in range(s_inter):
            s_b = s_func_cov(abs(para_contour[0, :, 0] - inter_contour[0, i, 0]))
            s_a_value = np.zeros([len(s_a), 1])
            s_a_value[0, 0] = 1
            for m in range(1, len(s_a)):
                s_a_value[m, 0] = s_a[m][i]

            for j in range(t_inter):
                t_b = t_func_cov(abs(para_contour[:, 0, 1] - inter_contour[j, 0, 1]))
                s_b = s_b.reshape([len(s_b), 1])
                t_b = t_b.reshape([len(t_b), 1])
                # So in surface_krig(), is there any similar error???????????????????????
                # If use 'sin', it will affect the results?
                t_a_value = np.zeros([len(t_a), 1])
                t_a_value[0, 0] = 1
                for m in range(1, len(t_a)):
                    t_a_value[m, 0] = t_a[m][j]
                s_vector = np.vstack((s_b, s_a_value))
                t_vector = np.vstack((t_b, t_a_value))
                for n in range(3):
                    inter_mat[j, i, n] = s_vector.T.dot(surf_mat[n]).dot(t_vector)
                # inter_mat[:, :, 4] = tow_code + 1  # Tow number
                # inter_mat[:, :, 3] = 1  # On the edge is 1, inside is 0.
        return inter_mat

    # def tow_boundary(self, center_x, center_y, boundary_mat, inter_x, name_drift, name_cov):
    # #    dis_list = np.sqrt((boundary_mat[0,:,0] - center_x)**2 +(boundary_mat[0,:,1] - center_y)**2)
    # #    theta_list = np.arctan2(center_y-boundary_mat[0,:,1], boundary_mat[0,:,0]-center_x)* 180 / np.pi
    #     dis_list = np.sqrt((boundary_mat[:,0] - center_x)**2 +(boundary_mat[:,1] - center_y)**2)
    # #    dis_list = np.sqrt((boundary_mat[:,0] - center_x)**2 +(boundary_mat[:,1] - center_y)**2)/30
    #     theta_list = np.arctan2(center_y-boundary_mat[:,1], boundary_mat[:,0]-center_x)
    # #    theta_list = np.arctan2(center_y-boundary_mat[:,1], boundary_mat[:,0]-center_x)* 180 / np.pi
    #
    #
    #     tot_list = pd.DataFrame(np.stack((dis_list, theta_list), axis = 1),columns=['dis','theta'])
    #     tot_list = tot_list.sort_values(by=['theta'])
    #
    #     func_b_para, func_a_para = one_dimensional_krig(tot_list, name_drift, name_cov)
    #
    #     plot_x = inter_x
    #     #plot_x = np.linspace(-3,3,100)
    #
    #
    #     func_drift, func_cov, len_a = func_select(name_drift, name_cov)
    #     adef=[1,1,1,1]
    #     krig_a = func_drift(inter_x,adef)
    #     len_b = tot_list.shape[0]
    #     krig_b = np.zeros([len_b, inter_x.shape[0]])
    #     for i in range(len_b):
    #         krig_b[i,:] = func_cov(abs(inter_x-tot_list['theta'].values[i])) # Here should be para_data[x,:,0] x for slice?????
    #
    #     if name_drift == 'lin':
    #         inter_y = (krig_a[0]*func_a_para[0]).repeat(plot_x.shape[0]).reshape(plot_x.shape[0],1) \
    #                  + (krig_a[1]*func_a_para[1]).reshape(plot_x.shape[0],1)\
    #                  + krig_b.T.dot(func_b_para)
    #     elif name_drift == 'quad' or 'trig':
    #         inter_y = (krig_a[0]*func_a_para[0]).repeat(plot_x.shape[0]).reshape(plot_x.shape[0],1) \
    #                  + (krig_a[1]*func_a_para[1]).reshape(plot_x.shape[0],1)+ (krig_a[2]*func_a_para[2]).reshape(plot_x.shape[0],1) \
    #                  + krig_b.T.dot(func_b_para)
    #
    #     inter_out = pd.DataFrame(np.stack((inter_x.reshape(-1), inter_y.reshape(-1)),axis = 1), columns = ['theta','dis'])
    #
    #     return inter_out
