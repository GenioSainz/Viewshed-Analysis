# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:34:19 2023

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm

from   scipy.interpolate import RegularGridInterpolator
from   viewShed_utils    import peaks,get_visible_area,view_sheed_vec,calc_distance

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
V  = view_sheed_vec(p1,X,Y,Z,Zi)

V_arr.append(V)
A_arr.append(get_visible_area(V))


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


plt.close('all')

Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Aspect = 1
px2inch  = 1/plt.rcParams['figure.dpi']
con_leve = np.arange(-7,9)

size_fig = (1200*px2inch,800*px2inch) 
# size_fig = (1000*px2inch,750*px2inch) 
fig      = plt.figure(figsize = size_fig,constrained_layout=True )

ax    = ['']*5
ax[0] = plt.subplot2grid((4,6),(0,0),colspan=4,rowspan=4)
ax[1] = plt.subplot2grid((4,6),(0,4),colspan=2)
ax[2] = plt.subplot2grid((4,6),(1,4),colspan=2)
ax[3] = plt.subplot2grid((4,6),(2,4),colspan=2)
ax[4] = plt.subplot2grid((4,6),(3,4),colspan=2)


ax[0].pcolormesh(X,Y,V,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)

CS = ax[0].contour(X,Y,Z,con_leve,colors='k',linewidths=0.75,zorder=3)
ax[0].clabel(CS, inline=True, fontsize=10)
ax[0].set_box_aspect(Aspect)
ax[0].set_xticks([])
ax[0].set_yticks([])

pts_x = np.array([0.0,-0.4, 1.2,1.5])
pts_y = np.array([1.5, 1.9,-0.1,0.3])

c = ['r','b','g','m']
s = ['A','B','C','D']
text_box = dict(boxstyle='round', facecolor=(0.9,0.9,0.9),alpha=1)

for i,(xi,yi) in enumerate (zip(pts_x ,pts_y)):
    
    xs,ys,zs,zt,dis = profile_view(Zi,p1,[xi,yi])
    
    ax[0].plot(xs,ys,'w',linestyle='-',linewidth=2)
    ax[0].plot(xs,ys,c[i],linestyle='--',linewidth=2)
    
    ax[0].text(xs[-1]+0.2,ys[-1]   ,s[i],va='center',ha='center', bbox=text_box,size=10)
    ax[0].text(xs[0]     ,ys[0]-0.2,'O' ,va='center',ha='center', bbox=text_box,size=10)
    
    ax[0].scatter(xs[0] ,ys[0] ,s=60,c='w',edgecolors='k',zorder=6)
    ax[0].scatter(xs[-1],ys[-1],s=60,c='w',edgecolors='k',zorder=6)
    
    ax[i+1].plot(dis,zs,c[i],label='Line of sight',linestyle='--')
    ax[i+1].plot(dis,zt,'k' ,label='Terrain')
    
    ax[i+1].text(dis[-1],zt[-1]-0.5,s[i],va='top'   ,ha='left'  , bbox=text_box,size=10)
    ax[i+1].text(dis[0] ,zs[0] +0.5,'O' ,va='bottom',ha='center', bbox=text_box,size=10)
    
    ax[i+1].set_xticklabels([]);ax[i+1].set_xticks([])
    ax[i+1].set_yticklabels([]);ax[i+1].set_yticks([])
    
    ax[i+1].legend(fontsize = 11)
    
plt.show()
plt.savefig('./imgs/terrain_slices.png',dpi = 150)



