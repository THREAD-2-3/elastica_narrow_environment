#!/usr/bin/env python
# coding: utf-8



import autograd as ag
import autograd.numpy as np
from scipy import optimize as op
from scipy import interpolate as ip
import matplotlib.pyplot as plt



import param
import rigid_wall_up as rw_up
import rigid_wall_down as rw_down
import mod_QVL as mod

param = param.param_all

from itertools import chain

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    # If arr is already one dimensional, just apply func1d directly
    if arr.ndim <= 1:
        return func1d(arr,  *args, **kwargs)
    
    # Extract shape and dimensions of arr for indexing
    dims = arr.shape
    Ni, Nk = dims[:axis],dims[axis+1:]
    # Obtain shape of output of func1d by running it through the first slice
    # along designated axis
    arr_view = arr.swapaxes(-1,axis)
    Nj_new = func1d(arr_view[(0,)*(len(arr_view.shape)-1)][:], *args, **kwargs).shape
    # Generate new array by applying the function using index magic
    func1d_arr = np.array([func1d(arr[ii + (slice(None,None,None),) + kk], *args, **kwargs) for ii in np.ndindex(Ni) for kk in np.ndindex(Nk)]).reshape( Ni + Nk + Nj_new )
    # Reorder to get the same ordering as arr. axis is now replaced by func1d_dim
    # new axes in its original position.
    order = tuple(range(len(func1d_arr.shape)))
    reordering_list = [order[:axis], order[axis:len(dims)-1], order[len(dims)-1:]]
    reordering_list[-1], reordering_list[1] = reordering_list[1], reordering_list[-1]
    return func1d_arr.transpose(tuple(chain.from_iterable(reordering_list)))


# spatial step
s_step = param.L/(param.n_nodes-1)


# Lagrangian function - second-order L for EB beam model
def Lagrangian(q,v,a,param):
    return 0.5 *param.EI* np.sum(a**2,axis=0) # axis=0 if the first dim gives me the dimension of the problem !!

def arclength(v,Lamdba,param):
    return Lamdba*(np.sum(v**2,axis=0)-1) # no axial strain


# discretization of the Lagrangian
def discrete_L(qk,vk,qkp1,vkp1,param):
    #Ferraro2021 with alpha = 1
    ak = ((-2*vkp1-4*vk)*s_step + 6*(qkp1-qk))/s_step**2
    akp1 = ((4*vkp1+2*vk)*s_step - 6*(qkp1-qk)) /s_step**2
    Ld_k = Lagrangian(qk,vk,ak,param)
    Ld_kp1 = Lagrangian(qkp1,vkp1,akp1,param)
    return 0.5*s_step*(Ld_k+Ld_kp1)

def discrete_arclength(vk,lambdak,param):
    return arclength(vk,lambdak,param)



# initial and final boundary conditions: [qx,qy] [vx,vy]
Q0 = np.array([0,0])
V0 = np.array([np.cos(np.pi/4),np.sin(np.pi/4)])

Qn = np.array([2,0])
Vn = np.array([np.cos(np.pi/4),np.sin(np.pi/4)]) 


# Initialize Lambda array
Lambda = np.zeros(param.n_nodes, order='F')

# Intialize QV array using spline interpolation
s = np.linspace(0,(param.n_nodes-1)*s_step,param.n_nodes) # Array of spatial coordinate
Qspline = ip.CubicSpline((0,(param.n_nodes-1)*s_step), (Q0,Qn), bc_type=((1,V0),(1,Vn)))
Vspline = Qspline.derivative()
QV = np.hstack((Qspline(s),Vspline(s))).transpose()

QVL = mod.joinQVL(QV,Lambda,rw_up.LambdaC1,rw_down.LambdaC2,param)



# new discrete Lagrangian with 1d-array input
def oneD_Ld(QVL_flat,param):
    QV = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)[0]
    Qk = QV[:,0,0:-1]
    Vk = QV[:,1,0:-1]
    Qkp1 = QV[:,0,1:]
    Vkp1 = QV[:,1,1:]
    return discrete_L(Qk,Vk,Qkp1,Vkp1,param)
    
def oneD_arclen(QVL_flat,param):
    QV, Lambda, rw_up.LambdaC1, rw_down.LambdaC2 = mod.separateQVL(mod.inflateQVL(QVL_flat, param, param.n_nodes), param)
    Vk = QV[:,1,1:-1] 
    Lambdak = Lambda[1:-1] 
    return discrete_arclength(Vk,Lambdak,param)



# complete action: sum of the Lds along the beam
def complete_Sd(QVL_flat, param):
    Ld_all = np.sum(apply_along_axis(oneD_Ld, 0, QVL_flat, param))
    gl_all = np.sum(oneD_arclen(QVL_flat, param))
    contact1_all = np.sum(rw_up.oneD_L_contact1(QVL_flat,param))
    contact2_all = np.sum(rw_down.oneD_L_contact2(QVL_flat,param))
    return Ld_all + gl_all + contact1_all + contact2_all 



def separateBC(QVL):
    return QVL[:, 1:-1], np.vstack((QVL[:, 0], QVL[:, -1])).transpose()

def joinBC(QVL1, QVLbound, param):
    return np.hstack((QVLbound[:,0, None], QVL1, QVLbound[:,-1, None]))



# Discrete Euler-Lagrange equations
def DEL(QVL1_flat, QVLbound_flat, param):
    QVL1 = mod.inflateQVL(QVL1_flat, param, param.n_nodes-2)
    QVLbound = mod.inflateQVL(QVLbound_flat, param, 2)
    return ag.grad(lambda y: complete_Sd(joinBC(y, QVLbound, param).flatten(order='F'),param))(QVL1).flatten(order='F')

# Jacobian of the discrete Euler-Lagrange equations
def DEL_automatic_jacobian(QVL1_flat, QVLbound_flat, param):
    return ag.jacobian(lambda y: DEL(y,QVLbound_flat,param))(QVL1_flat)



# Solution of the system
QVL1, QVLbound = separateBC(QVL)

sol = op.root(DEL,QVL1.flatten(order='F'),args=(QVLbound.flatten(order='F'),param),jac=DEL_automatic_jacobian)
QVL1_sol = mod.inflateQVL(sol.x, param, length=param.n_nodes-2)
print(sol.message)
# joining and reshaping the soluiton (q,v)
QVL_sol = joinBC(QVL1_sol, QVLbound, param)
QV_sol, Lambda_sol, LambdaC1_sol, LambdaC2_sol = mod.separateQVL(QVL_sol, param)


# trajectory
fig1, ax1 = plt.subplots(figsize=(13, 9))
ax1.plot(QVL[0,:],QVL[1,:],color='red',label='Initial guess - Spline')
ax1.plot(QVL_sol[0,:],QVL_sol[1,:],color='blue',label='solution')
ax1.plot(s,np.ones(param.n_nodes)*param.r1,'--',color='black')
ax1.plot(s,np.ones(param.n_nodes)*param.r2,'--',color='black')
ax1.set_aspect('equal')
ax1.legend(loc=8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.show()
# print distance from walls
gap1 = param.r1 - QVL_sol[1,:]
# print('gap first top wall',gap1)
gap2 = param.r2 - QVL_sol[1,:]
# print('gap second bottom wall',gap2)



# narrow tube
param = param._replace(r1 =0.45, r2=-0.45)

sol = op.root(DEL,QVL1_sol.flatten(order='F'),args=(QVLbound.flatten(order='F'),param),jac=DEL_automatic_jacobian)
QVL1_sol = mod.inflateQVL(sol.x, param, length=param.n_nodes-2)
print(sol.message)
QVL_sol_2 = joinBC(QVL1_sol, QVLbound, param)

# trajectory
fig1, ax1 = plt.subplots(figsize=(13, 9))
ax1.plot(QVL[0,:],QVL[1,:],color='red',label='Initial guess - Spline')
ax1.plot(QVL_sol[0,:],QVL_sol[1,:],color='blue',label='Partial solution r=||2||')
ax1.plot(QVL_sol_2[0,:],QVL_sol_2[1,:],color='green',label='Solution, r =||0.45||')
ax1.plot(s,np.ones(param.n_nodes)*param.r1,'--',color='black')
ax1.plot(s,np.ones(param.n_nodes)*param.r2,'--',color='black')
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
plt.savefig('contact.png', dpi=100, bbox_inches='tight')
plt.show()
# print distance from walls
gap1 = param.r1 - QVL_sol_2[1,:]
# print('gap first top wall',gap1)
gap2 = param.r2 - QVL_sol_2[1,:]
# print('gap second bottom wall',gap2)

# conctact forces
fig2, ax2 = plt.subplots()
ax2.plot(s,param.k*QVL_sol_2[-2,:],label='LambdaC_up')
ax2.plot(s,param.k*QVL_sol_2[-1,:],label='LambdaC_down')
plt.legend(loc='upper center')
ax2.set_xlabel('arc-length')
ax2.set_ylabel('contact forces')
plt.show()


