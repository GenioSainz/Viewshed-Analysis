# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:20:33 2023

@author: Genio
"""


import matplotlib.pyplot as plt
import numpy as np


def get_horizon(xo,ho,x,h):
    
        dh   = np.diff(h,prepend=h[0])
        ind1 = np.argwhere( (h>ho) & (dh>0) ).flatten()
        
        mh   = np.maximum.accumulate(h[ind1]).flatten()
        ind2 = np.argwhere( np.diff(mh,prepend=mh[0])>0 ).flatten()
    
        ind  = ind1[ind2]
        
        x_ind = x[ind]
        h_ind = h[ind]
        
        elevation   = np.arctan2(h_ind-ho,x_ind-xo) * 180/np.pi
        ind_max_ele = ind[np.argmax(elevation)]

        x_max_ele = x[ind_max_ele]
        h_max_ele = h[ind_max_ele]
        
        return x_max_ele,h_max_ele,elevation,x_ind,h_ind,x[ind1],h[ind1]
  
  
MS = 16

n  = 120
A  = np.linspace(1,3,n)**2
x  = np.linspace(np.pi,9*np.pi,n)
h  = A*np.cos(x)

    
plt.close('all')
px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (800*px2inch,400*px2inch) 
fig, ax  = plt.subplots(1,1,constrained_layout=True,figsize=size_fig)



xo = 4
ho = 2   
x_max_ele,h_max_ele,elevation,x_ind,h_ind,x_ind1,h_ind1= get_horizon(xo,ho,x,h)

m = (h_max_ele-ho)/(x_max_ele-xo)
r = ho+m*(x-xo)


ax0 = ax.twinx()
ax0.tick_params(axis='y', labelcolor='r')
ax0.scatter(x_ind,elevation,s=MS,color='r')

ax.scatter(x,h,s=MS)
ax.scatter(x_ind1,h_ind1,s=MS)
ax.scatter(x_ind ,h_ind ,s=2*MS,edgecolors='g',c='none')
ax.scatter(xo,ho,s=MS,color='k')

ax.plot(x,r,'k--')
ax.set_box_aspect(0.5)


plt.show()


