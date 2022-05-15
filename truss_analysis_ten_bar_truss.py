# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:24:42 2021

@author: nophibiton
"""

import numpy as np
import matplotlib.pyplot as plt

def define_truss_model():

    # set up material properties/younngs modulus
    E = np.ones([10,1]) * 30e3
    Area = np.ones([10,1]) * 10
    
    # set up the nodes
    nodes = []
    nodes.append([720, 360])
    nodes.append([720, 0])
    nodes.append([360, 360])
    nodes.append([360, 0])
    nodes.append([0, 360])
    nodes.append([0, 0])
    nodes = np.array(nodes).astype(float) # convert list into numpy array
    
    # set up element connectivity
    elems = []
    elems.append([4,2])
    elems.append([2,0])
    elems.append([5,3])
    elems.append([3,1])
    elems.append([2,3])
    elems.append([0,1])
    elems.append([4,3])
    elems.append([2,5])
    elems.append([2,1])
    elems.append([0,3])
    elems = np.array(elems).astype(int) # convert list into numpy array
    
    # define the loads
    loads = []
    loads.append([3,1,-100])
    loads.append([1,1,-100])
    loads = np.array(loads).astype(float)
    
    # set support displacements
    bcs = []
    bcs.append([4,0,0])
    bcs.append([4,1,0])
    bcs.append([5,0,0])
    bcs.append([5,1,0])
    bcs = np.array(bcs).astype(int)
    
    
    return nodes,elems,loads,E,Area,bcs


def Truss2DAnalysis():
    nodes,elems,loads,E,Area,bcs = define_truss_model()

    # get number of nodes, elements and bcs
    Nel, temp = elems.shape
    Nnodes, temp = nodes.shape
    Nbcs, temp = bcs.shape
    
    # initialize size of variables
    alldofs = np.arange(0,2*Nnodes)
    K = np.zeros([2*Nnodes,2*Nnodes]).astype(float)
    u = np.zeros([2*Nnodes,1]).astype(float)
    f = np.zeros([2*Nnodes,1]).astype(float)
    
    # determine index of specified dof
    dofspec = []
    for i in range(Nbcs):
        thisdof = 2*(bcs[i,0]) + bcs[i,1]
        dofspec.append(thisdof)
        u[thisdof] = bcs[i,2] # specify given displacement
    dofspec = np.array(dofspec).astype(int)    
    
    # determine index of free dof
    doffree = alldofs
    doffree = np.array(doffree)
    doffree = np.delete(doffree, dofspec, axis=0)
    
    # set-up the force vector
    Nloads, temp = loads.shape
    for i in range(Nloads):
        f[int(2*loads[i,0]+loads[i,1])] = loads[i,2]

    # initialize the global stiffness matrix
    for iel in range(Nel):
        elnodes = elems[iel,:]
        nodexy = nodes[elnodes,:]
        
        E1 = np.array([nodexy[1,0]-nodexy[0,0], nodexy[1,1]-nodexy[0,1] ])
        le = np.linalg.norm(E1) # get length of each element
        
        E1 = E1/le
        E2 = np.array([-1*E1[1],E1[0]])
        
        Kel_LOC = np.zeros([4,4])
        Kel_LOC [np.ix_([0,2],[0,2])]  = E[iel]*Area[iel]*np.array([[1,-1],[-1,1]])/le
        
        Qrot = np.array([E1,E2])
        r1 = np.c_[Qrot,np.zeros((2,2))]
        r2 = np.c_[np.zeros((2,2)),Qrot]
        Tmatrix = np.r_[r1,r2]
        
        Kel = np.dot(np.dot(Tmatrix.T , Kel_LOC),Tmatrix)
        
        eldofs = np.array([2*elnodes[0],2*elnodes[0]+1,2*elnodes[1],2*elnodes[1]+1])
        
        K[np.ix_(eldofs,eldofs)] = K[np.ix_(eldofs,eldofs)] + Kel
        
        
    kk = K[np.ix_(doffree,doffree)]
    ff = f[np.ix_(doffree)] - np.dot( K[np.ix_(doffree,dofspec)] , u[np.ix_(dofspec)])
        
    u[np.ix_(doffree)] = np.linalg.solve(kk,ff) 
        
    memforce = np.zeros([Nel,1])

    for iel in range(Nel):
        
        elnodes = elems[iel,:]
        nodexy = nodes[elnodes,:]
        
        E1 = np.array([nodexy[1,0]-nodexy[0,0], nodexy[1,1]-nodexy[0,1] ])
        le = np.linalg.norm(E1) # get length of each element
        
        E1 = E1/le
        E2 = np.array([-1*E1[1],E1[0]])
        
        Kel_LOC = np.zeros([4,4])
        Kel_LOC [np.ix_([0,2],[0,2])]  = E[iel]*Area[iel]*np.array([[1,-1],[-1,1]])/le
        
        Qrot = np.array([E1,E2])
        r1 = np.c_[Qrot,np.zeros((2,2))]
        r2 = np.c_[np.zeros((2,2)),Qrot]
        Tmatrix = np.r_[r1,r2]
        
        Kel = np.dot(np.dot(Tmatrix.T , Kel_LOC),Tmatrix)
        
        eldofs = np.array([2*elnodes[0],2*elnodes[0]+1,2*elnodes[1],2*elnodes[1]+1])
        
        uel = np.zeros([4,1]).astype(float)
        uel = u[np.ix_(eldofs)]
        
        fel = np.dot(Tmatrix, np.dot(Kel,uel))
            
        memforce[iel,0] = -fel[0]    
        
    return u,memforce,nodes,elems
    
def Plot2DTruss(nodes,c,lt,lw,lg):
    for i in range (len(elems)):
        xi, xf = nodes[elems[i,0],0] , nodes[elems[i,1],0]
        yi, yf = nodes[elems[i,0],1] , nodes[elems[i,1],1]
        line, = plt.plot([xi,xf],[yi,yf], color = c, linestyle = lt, linewidth = lw )
    line.set_label(lg)
    plt.legend(prop = {'size':8})
    
    
#%% Results    
u,memforce,nodes,elems = Truss2DAnalysis()

Plot2DTruss(nodes,'gray','--',1,'Undeformed')

Nnodes, temp = nodes.shape
u_deformed = np.zeros([Nnodes,2])
for i in range(Nnodes):
    u_deformed[i,0] = u[i]
    u_deformed[i,1] = u[i+1]
    
scale = 10
Dnodes = u_deformed*scale + nodes

Plot2DTruss(Dnodes,'red','-',2,'Deformed')

for i in range (0,len(u)):
    if (i+1) % 2 != 0:
        print("The x-displacement of node {} is {}".format(int(i/2+1),str(u[i])[1:-1] ))
    else:
        print("The y-displacement of node {} is {}".format(int((i-1)/2+1),str(u[i] )[1:-1] ))

for i in range(len(elems)):
    print("The stress on member {} is {}." .format(i+1, str(memforce[i])[1:-1] ))

