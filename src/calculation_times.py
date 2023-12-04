# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:32:25 2023

@author: Genio
"""

# %%
import time
import numpy  as np
import pandas as pd 

import matplotlib.pyplot as plt
from   matplotlib import cm

from scipy.interpolate import RegularGridInterpolator
from viewShed_utils import peaks,view_sheed_vec


def view_sheed_vec_times(p1,X,Y,Z,Zi,k=2):

    (nx, ny) = Z.shape
    n3Daxis = int(np.max([nx, ny])/k)  
    
    t0  = time.time()
    Aa  = np.zeros_like(X)            
    Bb  = np.ones_like(X)             
    T   = np.linspace(Aa, Bb, n3Daxis)
    t0e = time.time() - t0
    
    t1  = time.time()
    Xs  = p1[0] + T*(X-p1[0])    
    Ys  = p1[1] + T*(Y-p1[1])    
    Zs  = p1[2] + T*(Z-p1[2])    
    t1e = time.time() - t1
    
    t2  = time.time()
    xy  = np.stack((Xs.ravel(order='F'), Ys.ravel(order='F')),axis=1)
    Zt  = Zi(xy).reshape((n3Daxis, nx, ny), order='F')
    t2e = time.time() - t2
    
    t3  = time.time()
    con = Zt > Zs                    
    bol = np.sum(con, axis=0) > 2     
    V   = np.where(bol, np.nan, Z)    
    t3e = time.time() - t3
    
    
    times      = np.array([t0e, t1e, t2e, t3e])
    percents   = times/np.sum(times)
    time_array = np.stack((times,percents),axis=1)
    
    return V,time_array


npx  = 200
a, b = -3, 3
x, y = np.linspace(a, b, npx), np.linspace(a, b, npx)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = peaks(X, Y)
Zi = RegularGridInterpolator((x, y), Z,
                             method='linear',
                             bounds_error=False,
                             fill_value=None)
xp,yp  = -2,-2
view_h = 0.25

p1 = np.array([xp, yp, Zi((xp, yp))+view_h])

for k in [2,3,4]:
    
    V,time_array = view_sheed_vec_times(p1,X,Y,Z,Zi,k=k)
    lapse        = np.sum(time_array,axis=0)[0]
    
    indx = ['Create 3D Linspace',
             '3D Line p1p2 Ecuations',
             'Interpolation',
             'TerrainCoords>SkyCoords']
    
    cols =['Total Time', 'Time %']
    
    print(f'k factor={k}')
    print(pd.DataFrame(time_array,index=indx,columns=cols))
    print(f'Total Time = {lapse:0.5f}s \n')
    
# %%


npx = 300
a,b = -3,3
nx  = npx
ny  = npx
x,y = np.linspace(a,b,nx),np.linspace(a,b,ny)
X,Y = np.meshgrid(x,y,indexing='ij')
Z   = peaks(X,Y)
Zi  = RegularGridInterpolator((x,y),Z,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None)

xp = -2
yp = -2

view_h = 0.25
V_arr  = []
A_arr  = []
    
p1 = np.array([xp,yp,Zi((xp,yp))+view_h])


Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Ncon   = 25
Aspect = 1

plt.close('all')
px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (1000*px2inch,1000*px2inch) 
fig, ax  = plt.subplots(2,2,constrained_layout=True,figsize=size_fig)
data     = {'xlim':(a,b),'ylim':(a,b)}

for i,k in enumerate([2,3,4,5]):
    
   V  = view_sheed_vec(p1,X,Y,Z,Zi,k = k)

   ax = ax.flatten()
   con_leve = np.arange(-7,9)
   text_box = dict(boxstyle='round', facecolor='white',alpha=1)

       
   ax[i].pcolormesh(X,Y,V,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
   ax[i].scatter(p1[0],p1[1],s=60,c='w',edgecolors='k',zorder=6)
   CS = ax[i].contour(X,Y,Z,con_leve,colors='k',linewidths=0.75,zorder=3)
   ax[i].clabel(CS, inline=True, fontsize=10)
    
   ax[i].set_box_aspect(Aspect)
   ax[i].set(**data)
   ax[i].set_xticklabels([]);ax[i].set_xticks([])
   ax[i].set_yticklabels([]);ax[i].set_yticks([])

          
plt.show()


