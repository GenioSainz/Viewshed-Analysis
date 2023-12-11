# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:34:19 2023

@author: Genio
"""

from viewShed_utils import peaks,view_sheed_vec,calc_distance,get_visible_area

from   scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np
import time

import matplotlib.pyplot as plt
from   matplotlib import cm
import matplotlib.animation as animation


# Initialisation of the coordinate grid and the peaks surface
nframes = 150
view_h  = 0.5
a,b = -3,3
nxy = 250
x,y = np.linspace(a,b,nxy),np.linspace(a,b,nxy)
X,Y = np.meshgrid(x,y,indexing='ij')
Z   = peaks(X,Y)

# Surface grid interpolator
Zi  = RegularGridInterpolator((x,y),Z,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=None)


# Calculation of the observer's trajectory AB --> (xi,yi,zi)
px  = np.array([2.75 ,2.0 ,1.0 ,0.2 ,-0.6,-1.7,-2.2,-1.6,-0.5,-0.3,0.4,1.0,1.8,2.2,1.2,0.4,0.0,-1.5,-2.75])
py  = np.array([-2.75,-2.0,-2.0,-1.6,-2.5,-2.5,-2.0,-1.4,-0.6,0.0 ,0.4,0.8,1.0,1.5,2.2,2.0,1.6,2.0 ,2.75 ])
pts = np.array([px,py]).T

dis_norm = calc_distance(pts,norm=True)
t_norm   = np.linspace(0,1,nframes)

interp    = interp1d(dis_norm, pts, kind='cubic', axis=0)
inter_pts = interp(t_norm)
(xi,yi)   = inter_pts.T
zi        = Zi((xi,yi))+view_h

pts_profile = np.array([xi,yi,zi]).T
dis_profile = calc_distance(pts_profile,norm=False)

# Function which saves the animation data
def get_animation_data():
    V_arr = []
    
    for frame,(x_i,y_i,z_i) in enumerate((zip(xi,yi,zi))):
        
        p1 = np.array([x_i,y_i,z_i])  
        V  = view_sheed_vec(p1,X,Y,Z,Zi)
    
        V_arr.append(V)
        
        print(f'Frame: {frame+1}')
    
    np.save('./data/animation_data2.npy', V_arr) # save

#get_animation_data()


#%%
# PLOTS 
####################

# set defaults
plt.rcParams.update(plt.rcParamsDefault)

SMALL_SIZE  = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#  fonts
plt.rc('font',size=MEDIUM_SIZE)

# title
plt.rc('axes'  ,titlesize=MEDIUM_SIZE)
plt.rc('figure',titlesize=BIGGER_SIZE)

# xy-labells
plt.rc('axes',labelsize=SMALL_SIZE)

# xy-ticks
plt.rc('xtick',labelsize=SMALL_SIZE)
plt.rc('ytick',labelsize=SMALL_SIZE)

# legend
plt.rc('legend',fontsize = 13)
plt.rc('legend',facecolor='white')
plt.rc('legend',framealpha=0.9)

# lines
plt.rc('lines',linewidth=1.5)

plt.close('all')

# Load animation data
V_arr = np.load('./data/animation_data2.npy')
area  = [get_visible_area(V) for V in V_arr]

Cmin   = np.min(Z)
Cmax   = np.max(Z)
cmap   = cm.jet
Ncon   = 25
Aspect = 1

px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (900*px2inch,850*px2inch) 
fig, ax  = plt.subplots(1,1,constrained_layout=True,figsize=size_fig)
data     = {'xlabel':'$X _ (m)$','xlim':(a,b),
            'ylabel':'$Y _ (m)$','ylim':(a,b)}

# contour levels
CL = np.arange(-7,9)

# subplot(1,2,2)
###################
#quad1 = ax[0].pcolormesh(X,Y,Z,alpha=0.5,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=1)
quad2 = ax.pcolormesh(X,Y,V_arr[0],alpha=1,cmap=cmap,vmin=Cmin,vmax=Cmax,zorder=2)
cont1 = ax.contour(X,Y,Z,CL,colors='k',linewidths=0.75,zorder=3)
scat1 = ax.scatter(xi[0],yi[0],s=100,c='w',edgecolors='k',zorder=6)
line1 = ax.plot(xi,yi,ls='--',c='k')
cbar0 = fig.colorbar(quad2,fraction=0.05)
cbar0.ax.set_title('$Z _ (m)$')

ax.clabel(cont1, inline=True, fontsize=11)

text_box = dict(facecolor='k',edgecolor='w', alpha=1,boxstyle='round')     
ax.text(xi[0] ,yi[0], '$A$',c='w',weight='bold',zorder=10).set_bbox(text_box)
ax.text(xi[-1],yi[-1],'$B$',c='w',weight='bold',zorder=10).set_bbox(text_box)

ax.set_title(f'Visible Areas: {area[0]:6.3f} %')
ax.set_box_aspect(Aspect)
ax.set(**data)
ax.set_xticks([])
ax.set_yticks([])

# Run animation and save
run_animation   = True
save_animation  = True

if run_animation:

    def animate_fun(i):
        
        ax.set_title(f'Visible Area: {area[i]:6.3f} %')
        quad2.set_array(V_arr[i].ravel())
        scat1.set_offsets((xi[i],yi[i]))
        
        return quad2,scat1
    
    ani = animation.FuncAnimation(fig=fig,
                                    func=animate_fun,
                                    interval=100,
                                    frames=nframes)

if save_animation:
    
    writer = animation.PillowWriter(fps=10,
                                    metadata=dict(artist='Me'),
                                    bitrate=-1)
    
    t_start = time.time()
    
    ani.save('./imgs/animation_square.gif', writer=writer)
    
    t_end = time.time() - t_start 
    
    print(f'Lapse Time Write animation = {t_end:0.3f} s' )

plt.show()


