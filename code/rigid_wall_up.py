#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:24:01 2023

@author: martina
"""

import autograd.numpy as np

import mod_QVL as mod
import param

param = param.param_all

# Initialize LambdaC array
LambdaC1 = np.zeros(param.n_nodes, order='F')

# constraint Lagrangian
def distance_func1(q,LambdaC1,param): # activation of the unilateral constraint 
    g_constr = param.r1 - q
    xi = param.k*LambdaC1 - param.p*g_constr
    return np.minimum(xi,0)

def L_contact1(q,LambdaC1,param):
    g_constr = param.r1 - q
    return - param.k*np.dot(g_constr,LambdaC1) + 0.5*param.p*np.sum(g_constr**2,axis=0) - 0.5/param.p*np.sum((distance_func1(q,LambdaC1,param))**2)

# discretisation
def discrete_distance_func1(qk,LambdaCk,param):
    return distance_func1(qk,LambdaCk,param)

def discrete_L_contact1(qk,LambdaCk,param):
    return L_contact1(qk,LambdaCk,param)

# new discrete Lagrangian with 1d-array input
def oneD_distance_func1(QVL_flat,param):
    # Inflate and separate QVL
    QV, Lambda, LambdaC1, LambdaC2 = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)
    Qyk = QV[1,0,1:-1]
    LambdaC1k = LambdaC1[1:-1]
    return discrete_distance_func1(Qyk,LambdaC1k,param)

def oneD_L_contact1(QVL_flat,param):
    # Inflate and separate QVL
    QV, Lambda, LambdaC1, LambdaC2 = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)
    Qyk = QV[1,0,1:-1]
    LambdaC1k = LambdaC1[1:-1]
    return discrete_L_contact1(Qyk,LambdaC1k,param)

