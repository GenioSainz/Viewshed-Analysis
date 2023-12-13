# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:28:04 2023

@author: Genio
"""


import numpy as np
import matplotlib.pyplot as plt



def fxy(X,Y):
    
    return X**2 + Y**2 + np.where((X>0) & (Y>0),1,0)

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,50)
X,Y = np.meshgrid(x,y)
Z =  fxy(X,Y)

# plt.close('all')
# px2inch  = 1/plt.rcParams['figure.dpi']
# size_fig = (400*px2inch,400*px2inch) 
# fig, ax  = plt.subplots(subplot_kw={"projection": "3d"})

# ax.set(xlabel='X',ylabel='Y')


# ax.plot_surface(X,Y,Z)

# fig.show()

# x = np.array([1,2,3,4,5])
# y = np.array([6,7,8,9])

# X,Y = np.meshgrid(x,y,indexing='xy')
# print(X),print(Y),print('\n')

# X,Y = np.meshgrid(x,y,indexing='ij')
# print(X),print(Y),print('\n')




