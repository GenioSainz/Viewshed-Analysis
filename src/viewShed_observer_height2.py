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
from   view_shed_utils   import peaks,get_visible_area,view_sheed_vec,get_visible_area,calc_distance

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


xi = -2
yi = -2

view_h = 0.25
V_arr  = []
A_arr  = []
    
p1 = np.array([xi,yi,Zi((xi,yi))+view_h])
V  = view_sheed_vec(p1,X,Y,Z,Zi)

V_arr.append(V)
A_arr.append(get_visible_area(V))



#%% PLOTS 



def profile_view(Zi,p1,p2):

    
    p2 = np.array([p2[0],p2[1],Zi((p2[0],p2[1]))])
    n  = 100
    xs = np.linspace(p1[0],p2[0],n)
    ys = np.linspace(p1[1],p2[1],n)
    zs = np.linspace(p1[2],p2[2],n)
    
    pts = np.stack([xs,ys],axis=1) 
    zt  = Zi(pts)   
    dis   = calc_distance(pts,norm=False)
    
    return xs,ys,zs,zt,dis



xs,ys,zs,zt,dis = profile_view(Zi,p1,[2,2])



plt.close('all')

Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Aspect = 1

px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (1600*px2inch,800*px2inch) 
fig, ax  = plt.subplots(1,2,constrained_layout=True,figsize=size_fig)
data     = {'xlim':(a,b),'ylim':(a,b)}

ax = ax.flatten()
con_leve = np.arange(-7,9)
text_box = dict(boxstyle='round', facecolor='white',alpha=1)

ax[0].pcolormesh(X,Y,V,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
ax[0].scatter(xi,yi,s=60,c='w',edgecolors='k',zorder=6)
CS = ax[0].contour(X,Y,Z,con_leve,colors='k',linewidths=0.75,zorder=3)
ax[0].clabel(CS, inline=True, fontsize=10)

ax[0].plot(xs,ys)

ax[0].set_box_aspect(Aspect)
ax[0].set(**data)


ax[1].plot(dis,zs)
ax[1].plot(dis,zt)




    
plt.show()
# plt.savefig('../imgs/observer_position.png',dpi = 150)



