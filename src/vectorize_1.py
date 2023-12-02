# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:06:25 2023

@author: Genio
"""
from   scipy.interpolate import RegularGridInterpolator
import numpy as np
import time

p = np.array([3,4,1])

px,py,pz = p


x   = np.array([1,2])
y   = np.array([5,6])
X,Y = np.meshgrid(x,y)
Z   = np.arange(4).reshape(2,-1)

Zi  = RegularGridInterpolator((x,y),Z,
                                      method='linear',
                                      bounds_error=False,
                                      fill_value=None)

n     = 3
nx,ny = Z.shape

a = np.zeros_like(X)
b = np.ones_like(X)
T = np.linspace(a,b,n)

Xr = px+T*(X-px)
Yr = py+T*(Y-py)
Zs = pz+T*(Z-pz)

xy  = np.vstack((Xr.ravel(order='F'),Yr.ravel(order='F'))).T
Zt  = Zi(xy).reshape((n,nx,ny),order='F')
con = Zt.ravel(order='F')>Zs.ravel(order='F')
con = con.reshape((3,nx,ny),order='F')
bol = np.sum(con,axis=0)>2
V   = np.where(bol,1,0)


