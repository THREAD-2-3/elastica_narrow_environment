#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:34:56 2023

@author: martina
"""

import autograd.numpy as np

import mod_QVL as mod
import param

param = param.param_all

# Initialize LambdaC array
LambdaC2 = np.zeros(param.n_nodes, order='F')

# constraint Lagrangian
def distance_func2(q,LambdaC2,param): # activation of the unilateral constraint 
    g_constr = q - param.r2 
    xi = param.k*LambdaC2 - param.p*g_constr
    return np.minimum(xi,0)

def L_contact2(q,LambdaC2,param):
    g_constr = q - param.r2 
    return - param.k*np.dot(g_constr,LambdaC2) + 0.5*param.p*np.sum(g_constr**2,axis=0) - 0.5/param.p*np.sum((distance_func2(q,LambdaC2,param))**2)

# discretisation
def discrete_distance_func2(qk,LambdaCk,param):
    return distance_func2(qk,LambdaCk,param)

def discrete_L_contact2(qk,LambdaCk,param):
    return L_contact2(qk,LambdaCk,param)

# new discrete Lagrangian with 1d-array input
def oneD_distance_func2(QVL_flat,param):
    # Inflate and separate QVL
    QV, Lambda, LambdaC1, LambdaC2 = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)
    Qyk = QV[1,0,1:-1]
    LambdaC2k = LambdaC2[1:-1]
    return discrete_distance_func2(Qyk,LambdaC2k,param)

def oneD_L_contact2(QVL_flat,param):
    # Inflate and separate QVL
    QV, Lambda, LambdaC1, LambdaC2 = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)
    Qyk = QV[1,0,1:-1]
    LambdaC2k = LambdaC2[1:-1]
    return discrete_L_contact2(Qyk,LambdaC2k,param)


