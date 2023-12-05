# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:31:22 2023

@author: Genio
"""

import matplotlib.pyplot as plt
from   matplotlib import cm
import numpy as np

from   scipy.interpolate import RegularGridInterpolator
from   viewShed_utils   import peaks,view_sheed_vec

# Initialisation of the coordinate grid and the peaks surface
npx = 200
a,b = -3,3
nx  = npx
ny  = npx
x,y = np.linspace(a,b,nx),np.linspace(a,b,ny)
X,Y = np.meshgrid(x,y,indexing='ij')
Z   = peaks(X,Y)

# Surface grid interpolator
Zi  = RegularGridInterpolator((x,y),Z,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None)

xp = -2
yp = -2

view_h = 1.5
p1     = np.array([xp,yp,Zi((xp,yp))+view_h])
V      = view_sheed_vec(p1,X,Y,Z,Zi)


plt.close('all')

cmap   = cm.jet
Cmin   = np.min(Z)
Cmax   = np.max(Z)
elev   = 35
azim   = -145

px2inch  = 1/plt.rcParams['figure.dpi']
con_leve = np.arange(-7,9)
size_fig = (1400*px2inch,700*px2inch) 
fig, ax  = plt.subplots(1,2,
                           constrained_layout=True,
                           figsize=size_fig,
                           subplot_kw={'projection':'3d','computed_zorder':False})

egc = 'gray'
lw  = 0.5
aspect = (1,1,1)
rcc=35
ax[0].plot_surface(X,Y,Z,cmap=cmap,vmin=Cmin,vmax=Cmax,edgecolors=egc,linewidth=lw,rcount=rcc,ccount=rcc)
ax[0].view_init(elev=elev, azim=azim)

ax[0].set_xlim(a,b),ax[0].set_ylim(a,b),ax[0].set_zlim(Cmin,Cmax)
ax[0].tick_params(colors='w'),ax[0].set_box_aspect(aspect)

ax[1].plot_surface(X,Y,Z,color='w',alpha=0.6,edgecolors='w',linewidth=lw,zorder=5,rcount=2*rcc,ccount=2*rcc)
ax[1].plot_surface(X,Y,V,cmap=cmap,vmin=Cmin,vmax=Cmax,edgecolors=egc,linewidth=lw,zorder=10,rcount=rcc,ccount=rcc)
ax[1].scatter(p1[0],p1[1],p1[2],s=60,c='w',edgecolors='k',zorder=20)

ax[1].view_init(elev=elev, azim=azim)
ax[1].set_xlim(a,b),ax[1].set_ylim(a,b),ax[1].set_zlim(Cmin,Cmax)
ax[1].tick_params(colors='w'),ax[1].set_box_aspect(aspect)



from matplotlib.lines   import Line2D
from matplotlib.patches import Patch

p1 = Line2D([np.nan],[np.nan],marker='o',color='w', mfc='w',mec='k')
p2 = Patch(facecolor='gray', edgecolor='gray',alpha=0.5)

plots  = [p1,p2]
labels = ['Observer Position', 'Not Visible Areas']
fig.legend(plots, labels,loc='upper center',fontsize=14)
   
plt.show()
plt.savefig('../imgs/3D_comparison.png',dpi = 150)