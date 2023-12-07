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
              [xn,yn,zn]]
    
       if norm, return normalised distance [0...1]
    '''
       
    diff     = np.diff(pts,axis=0)**2
    distance = np.cumsum( np.sqrt( np.sum(diff,axis=1) ))
    distance = np.insert(distance,0,0)
    
    if norm:
        distance = distance/ distance[-1]
    
    return distance


def view_sheed_vec(p1,X,Y,Z,Zi,k=2):
    ''' Vectorised implementation of viewSheed
    
        Inputs:
        p1 : Observer 3D point
        X,Y: 2D coordinates matrix
        Z  : 2D surface matrix
        Zi : scipy surface interpolator zi = Zi((xi,yi))
        k  : size factor of the 3D axis 
        
        Output:
        V : Visibility matrix of shape equal to Z
        V values:
          Points(i,j) of Z visbles from p1     : Z(i,j)
          Points(i,j) of Z not visible from p1 : np.nan
    '''
    
    t_start = time.time()
    
    (nx,ny) = Z.shape
    n3Daxis = int(np.max([nx,ny])/k) # Size of the arrays 3D axis 
    
    # T Parametric matrix
    Aa = np.zeros_like(X)            # 2D float array (nx,ny)
    Bb = np.ones_like(X)             # 2D float array (nx,ny)
    T  = np.linspace(Aa,Bb,n3Daxis)  # 3D float array 
    
    # Xs,Ys,Zs store the 3D coordinates of the line joining 
    # each point on the surface to the observer's position
    # x = x0 + t(x1-x0)
    # y = y0 + t(y1-y0)
    # z = z0 + t(z1-z0)
    Xs = p1[0] + T*(X-p1[0])    # 3D float array
    Ys = p1[1] + T*(Y-p1[1])    # 3D float array
    Zs = p1[2] + T*(Z-p1[2])    # 3D float array
    
    xy  = np.stack((Xs.ravel(order='F'),Ys.ravel(order='F')),axis=1)  # 2D float array
    Zt  = Zi(xy).reshape((n3Daxis,nx,ny),order='F')                   # 3D float array
    
    con = Zt>Zs                     # 3D bool array
    bol = np.sum(con,axis=0)>2      # 2D float array
    V   = np.where(bol,np.nan,Z)    # 2D float array
    
    t_end = time.time() - t_start 
    
    print(f'ViewShed Vectorize ({nx},{ny}) Lapse Time = {t_end:0.4f} s' )
    
    return V    


def view_sheed_for(p1,X,Y,Z,Zi,k=2):
    '''Not vectorized view_sheed implementation
    '''
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


def view_sheed(p1,x,y,Z,h=0.5):
    '''Initial non-vectorised implementation of view_sheed by varying
    the number of points to be evaluated in each iteration based on the
    distance and size of each cell.'''
    
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
