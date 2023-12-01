# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:52:59 2023

@author: Genio
"""
from   scipy.interpolate import RegularGridInterpolator
import numpy as np
import time 

def peaks(X,Y):
    '''Matlab Style Peaks Function '''
    
    k1 = 3*(1-X)**2 * np.exp(-(X**2) - (Y+1)**2) 
    k2 = - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)
    k3 = - 1/3*np.exp(-(X+1)**2 - Y**2)
    
    return  k1+k2+k3


def roundGrid(x,cellSize):
    '''Round coordinates to regular grid spaced cellSize'''
    return np.round(x/cellSize)*cellSize


def get_visible_area(V):
     '''Return the percentage of the visible area with 
        respect to the total area'''
     (rows,cols) = np.shape( np.argwhere(np.isnan(V)) )
     
     return 100*(1-rows/(V.size))
 
 
def calc_distance(pts,norm=False):
    '''pts = [[x1,y1,z1],
              [x2,y2,z2]
              [xn,yn,zn]]'''
       
    diff       = np.diff(pts,axis=0)**2
    distance   = np.cumsum( np.sqrt( np.sum(diff,axis=1) ))
    distance   = np.insert(distance,0,0)
    
    if norm:
        distance =  distance/ distance[-1]
    
    return distance
    
    
def view_sheed(p1,x,y,Z,h=0.5):
    
    t_start = time.time()
    
    nx  = x.size
    ny  = y.size
    
    X,Y = np.meshgrid(x,y,indexing='ij')
    Zi  = RegularGridInterpolator((x,y),Z,
                                          method='linear',
                                          bounds_error=False,
                                          fill_value=None)
    
    dxy   = np.diff(x)[0]
    d2D   = np.sqrt( (X-p1[0])**2 + (Y-p1[1])**2  )
    Npx   = (np.ceil(d2D/dxy)+1).astype(int)
    
    p1    = roundGrid(p1,dxy)
    p1[2] = Zi((p1[0],p1[1])) + h
    print(p1)
    
    X_   = X.flatten(  order='F')
    Y_   = Y.flatten(  order='F')
    Z_   = Z.flatten(  order='F')
    Npx_ = Npx.flatten(order='F')
    V    = Z_
    
    for i,(xi,yi,zi,npx) in enumerate(zip(X_,Y_,Z_,Npx_)):
        
            p2 = np.array([xi,yi,zi])
            v  = p2-p1
        
            t    = np.linspace(0,1,npx).reshape(-1,1)
            xr   = p1[0] + t*v[0]
            yr   = p1[1] + t*v[1]
            zSky = p1[2] + t*v[2]
            
            points = np.hstack((xr,yr))
            zTer   = Zi(points).reshape((-1,1))
            
            if np.sum(zTer>zSky)>2:
                V[i]=np.nan
            
           
    V = np.reshape(V,(nx,ny),order='F')  

    t_end = time.time() - t_start 
    print(f'nx={nx} ny={ny} LapseTime={t_end:0.4f} s' )
    
    return V,p1


def view_sheed_vec(p1,X,Y,Z,Zi,k=2):
    
    t_start = time.time()
    
    (nx,ny) = Z.shape
    
    n  = int(np.max([nx,ny])/k)
    Aa = np.zeros_like(X)
    Bb = np.ones_like(X)
    T  = np.linspace(Aa,Bb,n)
    
    Xs = p1[0] + T*(X-p1[0])
    Ys = p1[1] + T*(Y-p1[1])
    Zs = p1[2] + T*(Z-p1[2])
    
    xy  = np.stack((Xs.ravel(order='F'),Ys.ravel(order='F')),axis=1)
    Zt  = Zi(xy).reshape((n,nx,ny),order='F')
    
    con = Zt.ravel(order='F') > Zs.ravel(order='F')
    con = con.reshape((n,nx,ny),order='F')
    bol = np.sum(con,axis=0)>2
    V   = np.where(bol,np.nan,Z)
    
    t_end = time.time() - t_start 
    
    print(f'ViewShed Vectorize ({nx},{ny}) Lapse Time = {t_end:0.4f} s' )
    
    return V


def view_sheed_for(p1,X,Y,Z,Zi,k=2):
    
    t_start = time.time()
    
    (nx,ny) = Z.shape
    
    n  = int(np.max([nx,ny])/k)
    
    X_ = X.flatten(order='F')
    Y_ = Y.flatten(order='F')
    Z_ = Z.flatten(order='F')
    V  = Z_
    t  = np.linspace(0,1,n).reshape(-1,1)
    
    for i,(xi,yi,zi) in enumerate(zip(X_,Y_,Z_)):
        
            p2 = np.array([xi,yi,zi])
            v  = p2-p1
        
            xr   = p1[0] + t*v[0]
            yr   = p1[1] + t*v[1]
            zSky = p1[2] + t*v[2]
            
            points = np.hstack((xr,yr))
            zTer   = Zi(points).reshape((-1,1))
            
            if np.sum(zTer>zSky)>2:
                V[i]=np.nan
            
           
    V = np.reshape(V,(nx,ny),order='F')  

    t_end = time.time() - t_start 
    print(f'ViewShed Foor Loop ({nx},{ny}) Lapse Time = {t_end:0.3f} s' )
    
    return V