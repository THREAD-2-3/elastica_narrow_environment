#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:51:50 2023

@author: martina
"""

import param
import autograd.numpy as np

param = param.param_all

def inflateQVL(QVL_flat, param, length=param.n_nodes):
    return QVL_flat.reshape(param.n_dim*2 + 1 +1 +1, length, order='F')

def separateQVL(QVL, param):
    # QVL is an (n_dim*2 + 1 +1, n_nodes) array: (qk vk lambdak lambdaCk) is the k-th column.
    # This separates it into three arrays, QV (n_dim, 2, n_nodes) [dimension, q or v, node]
    # Lambda (n_nodes) and LambdaC (n_nodes). Using Fortran ordering, there is no need to transpose.
    return (QVL[0:-3,:].reshape(param.n_dim,2,param.n_nodes,order='F'), QVL[-3,:], QVL[-2,:], QVL[-1,:])

def joinQVL(QV, Lambda, LambdaC1, LambdaC2, param):
    # This joins QV (n_dim, 2, n_nodes), Lambda (n_nodes) and LambdaC (n_nodes) into a QVL
    # (n_dim*2 + 1 +1, n_nodes) array. Using Fortran ordering, there is no need to transpose.
    return np.vstack((QV.reshape(param.n_dim*2,param.n_nodes,order='F'),Lambda,LambdaC1,LambdaC2))

