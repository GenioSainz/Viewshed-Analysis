# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:06:25 2023

@author: Genio
"""
import matplotlib.pyplot as plt
from   matplotlib import cm
from   scipy.interpolate import RegularGridInterpolator
from   view_shed_utils   import peaks,view_sheed_for,view_sheed_vec,get_visible_area
import numpy as np
import time



nframes=200

px,py = -2,-2


a,b = -3,3
h   = 0.5
nx  = nframes
ny  = nframes

x,y = np.linspace(a,b,nx),np.linspace(a,b,ny)
X,Y = np.meshgrid(x,y,indexing='ij')
Z   = peaks(X,Y)
Zi  = RegularGridInterpolator((x,y),Z,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None)

p  = np.array([px,py,Zi((px,py))+h])

V1 = view_sheed_vec(p,X,Y,Z,Zi)
V2 = view_sheed_for(p,X,Y,Z,Zi)

area1  = get_visible_area(V1)
area2  = get_visible_area(V2)

Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Ncon   = 25
Aspect = 1

plt.close('all')
px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (1800*px2inch,900*px2inch) 
fig, ax  = plt.subplots(1,2,constrained_layout=True,figsize=size_fig)
data     = {#'title':'ViewShed',
            'xlabel':'$X$ [m]','xlim':(a,b),
            'ylabel':'$Y$ [m]','ylim':(a,b)}


# subplot(1,2,2)
###################
quad1 = ax[0].pcolormesh(X,Y,Z,alpha=0.3,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
quad2 = ax[0].pcolormesh(X,Y,V1,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
cont1 = ax[0].contour(X,Y,Z,Ncon,colors='k',linewidths=0.75,zorder=3)
scat1 = ax[0].scatter(p[0],p[1],s=60,c='r',edgecolors='k',zorder=6)
cbar0 = fig.colorbar(quad2,ax=ax[0],fraction=0.05)

ax[0].set_title(f'Visible Area: {area1:6.3f} %')
ax[0].set_box_aspect(Aspect)
ax[0].set(**data)

# subplot(1,2,2)
###################
quad1 = ax[1].pcolormesh(X,Y,Z,alpha=0.3,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
quad2 = ax[1].pcolormesh(X,Y,V2,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
cont1 = ax[1].contour(X,Y,Z,Ncon,colors='k',linewidths=0.75,zorder=3)
scat1 = ax[1].scatter(p[0],p[1],s=60,c='r',edgecolors='k',zorder=6)
cbar0 = fig.colorbar(quad2,ax=ax[0],fraction=0.05)

ax[1].set_title(f'Visible Area: {area2:6.3f} %')
ax[1].set_box_aspect(Aspect)
ax[1].set(**data)

plt.show()