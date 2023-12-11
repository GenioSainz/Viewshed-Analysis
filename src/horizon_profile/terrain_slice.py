# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:20:33 2023

@author: Genio
"""


import matplotlib.pyplot as plt
import numpy as np

# PLOTS 
####################

# set defaults
plt.rcParams.update(plt.rcParamsDefault)

SMALL_SIZE  = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#  fonts
plt.rc('font',size=MEDIUM_SIZE)

# title
plt.rc('axes'  ,titlesize=MEDIUM_SIZE)
plt.rc('figure',titlesize=BIGGER_SIZE)

# xy-labells
plt.rc('axes',labelsize=MEDIUM_SIZE)

# xy-ticks
plt.rc('xtick',labelsize=SMALL_SIZE)
plt.rc('ytick',labelsize=SMALL_SIZE)

# legend
plt.rc('legend',fontsize = SMALL_SIZE)
plt.rc('legend',facecolor='white')
plt.rc('legend',framealpha=0.5)

# lines
plt.rc('lines',linewidth=1.5)

plt.close('all')

def get_horizon(xo,ho,x,h):
    
        dh   = np.diff(h,prepend=h[0])
        ind1 = np.argwhere( (h>ho) & (dh>0) ).flatten()
        
        mh   = np.maximum.accumulate(h[ind1]).flatten()
        ind2 = ind1[ np.argwhere( np.diff(mh,prepend=mh[0])>0 ).flatten() ]
        
        elevation_angle = np.arctan2(h[ind2]-ho,x[ind2]-xo) * 180/np.pi
        ind_max_ele     = ind2[np.argmax(elevation_angle)]

        x_horizon = x[ind_max_ele]
        h_horizon = h[ind_max_ele]
        
        slope = (h_horizon-ho)/(x_horizon-xo)
        LOS   = ho+slope*(x-xo)
        
        return x_horizon,h_horizon,elevation_angle,ind1,ind2,LOS
  
  
MS = 50
ML = MS*2.5

n  = 90
A  = np.linspace(1,3,n)**2
x  = np.linspace(0,5*np.pi,n)
h  = A*np.sin(x)


px2inch  = 1/plt.rcParams['figure.dpi']
size_fig = (1600*px2inch,800*px2inch) 
fig, ax  = plt.subplots(1,1,constrained_layout=True,figsize=size_fig)


xo = 0
ho = 1
x_horizon,h_horizon,elevation_angle,ind1,ind2,LOS = get_horizon(xo,ho,x,h)

ax.scatter(xo,ho              ,s=MS,label='Observer Position')
ax.scatter(x,h                ,s=MS,label='Terrain Profile')
ax.scatter(x[ind1],h[ind1]    ,s=MS,label='Pts1 = h>ho and Diff(h)>0')
ax.scatter(x[ind2],h[ind2]    ,s=ML,label='Pts2 = Cummax(pts1) and Diff( Cummax(pts1) )>0',marker='s',c='none',edgecolors='k',linewidths=1.5)
ax.scatter(x_horizon,h_horizon,s=ML,label='Horizon Point',marker='s',c='none',edgecolors='r',linewidths=1.5)
ax.plot(x,LOS,'r--'                  ,label='Line Of Sight',zorder=0)
ax.scatter(np.nan,np.nan,s=MS,label='Elevation Angle From Observer Position To Pts2',c='m')

right_ax_color = '#ff7f0e'
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Terrain Altitude [m]',color=right_ax_color)
ax.tick_params(axis='y', labelcolor=right_ax_color)
ax.legend(loc='upper left')

ax_right = ax.twinx()
ax_right.tick_params(axis='y', labelcolor='m')
ax_right.scatter(x[ind2],elevation_angle,s=MS,c='m')
ax_right.set_ylim(5,30)
ax_right.set_ylabel('Elevation angle [º]',color='m')


plt.show()


