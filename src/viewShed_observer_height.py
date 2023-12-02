# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:19:28 2023

@author: Genio
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:34:19 2023

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm

from   scipy.interpolate import RegularGridInterpolator
from   view_shed_utils   import peaks,get_visible_area,view_sheed_vec,get_visible_area

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


xy    = np.array([-2,0,2])
Xi,Yi = np.meshgrid(xy,xy,indexing='xy')
Xi    = Xi.flatten()
Yi    = np.flip(Yi.flatten())


view_h = 2.5
V_arr  = []
A_arr  = []

for xi,yi in zip(Xi,Yi):
    
    p1 = np.array([xi,yi,Zi((xi,yi))+view_h])
    V  = view_sheed_vec(p1,X,Y,Z,Zi)
    
    V_arr.append(V)
    A_arr.append(get_visible_area(V))



#%% PLOTS 

   
plt.close('all')


Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Ncon   = 25
Aspect = 1

px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (1000*px2inch,1000*px2inch) 
fig, ax  = plt.subplots(3,3,constrained_layout=True,figsize=size_fig)
data     = {'xlim':(a,b),'ylim':(a,b)}

ax = ax.flatten()
con_leve = np.arange(-7,9)
text_box = dict(boxstyle='round', facecolor='white',alpha=1)

for i,(vi,ai) in enumerate(zip(V_arr,A_arr)):
    
    ax[i].pcolormesh(X,Y,vi,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
    ax[i].scatter(Xi[i],Yi[i],s=60,c='w',edgecolors='k',zorder=6)
    CS = ax[i].contour(X,Y,Z,con_leve,colors='k',linewidths=0.75,zorder=3)
    ax[i].clabel(CS, inline=True, fontsize=10)
    
    ax[i].set_box_aspect(Aspect)
    ax[i].set(**data)
    ax[i].set_xticklabels([]);ax[i].set_xticks([])
    ax[i].set_yticklabels([]);ax[i].set_yticks([])

    
    textstr = ' '.join([f'H:{view_h:2.1f}m',f'A:{ai:5.2f}%'])
    ax[i].text(0.05, 0.075, textstr, transform=ax[i].transAxes, weight='bold',va='top',fontsize=8,ha='left', bbox=text_box)
    
plt.show()
plt.savefig('../imgs/observer_position.png',dpi = 150)



