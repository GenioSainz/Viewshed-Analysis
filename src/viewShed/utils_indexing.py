# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:28:04 2023

@author: Genio
"""


import numpy as np
from   scipy.interpolate import RegularGridInterpolator

def ndtotext(A, w=None, h=None):
    if A.ndim==1:
        if w == None :
            return str(A)
        else:
            s= '['
            for i,AA in enumerate(A[:-1]):
                s += str(AA)+' '*(max(w[i],len(str(AA)))-len(str(AA))+1)
            s += str(A[-1])+' '*(max(w[-1],len(str(A[-1])))-len(str(A[-1]))) +'] '
    elif A.ndim==2:
        w1 = [max([len(str(s)) for s in A[:,i]])  for i in range(A.shape[1])]
        w0 = sum(w1)+len(w1)+1
        s= u'\u250c'+u'\u2500'*w0+u'\u2510' +'\n'
        for AA in A:
            s += ' ' + ndtotext(AA, w=w1) +'\n'
        s += u'\u2514'+u'\u2500'*w0+u'\u2518'
    elif A.ndim==3:
        strings=[ndtotext(a)+'\n' for a in A]
        s='\n'.join(''.join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s


x = np.array([0,1,2,4])
y = np.array([5,6,7])

Xc,Yc = np.meshgrid(x,y,indexing='xy')
Xm,Ym = np.meshgrid(x,y,indexing='ij')

print('xy indexing (yn,xn)')
print('Xc',' '*10,'Yc')
print(ndtotext(np.stack([Xc,Yc])))

print('\n')

print('ij indexing (xn,yn)')
print('Xm',' '*10,'Ym')
print(ndtotext(np.stack([Xm,Ym])))


n3Daxis = 3
x = np.array([1,2,3])
y = np.array([4,5,6,7])
X,Y = np.meshgrid(x,y,indexing='ij')
Z   = np.random.randint(2,7,size=X.shape)
Zinterp = RegularGridInterpolator((x,y),Z,
                                        method='linear',
                                        fill_value=None,
                                        bounds_error=False)

x0 = 2.5
y0 = 6.5
h0 = 0.25
p0 = np.array([x0,y0,Zinterp((x0,y0))+h0])

# x = x0 + t(x1-x0) x1->X
# y = y0 + t(y1-y0) y1->Y 
# z = z0 + t(z1-z0) z1->Z
# Broadcasting Vectorization
t    = np.linspace(0,1,n3Daxis).reshape(n3Daxis,1,1)
Xlos = p0[0] + t*(X-p0[0])
Ylos = p0[1] + t*(Y-p0[1])
Zlos = p0[2] + t*(Z-p0[2])

Zter = Zinterp(( Xlos.ravel(order='F'),Ylos.ravel(order='F') ))
Zter = Zter.reshape(Zlos.shape,order='F')
V    = np.where( np.sum(Zter>Zlos,axis=0)>1, np.nan, Z)

print('p0',p0.shape)
print(ndtotext(p0),'\n')

print('t',t.shape)
print(ndtotext(t),'\n')

print('X,Y,Z',X.shape)
print(ndtotext(np.stack([X,Y,Z])),'\n')

print('Xlos',Xlos.shape)
print(ndtotext(Xlos),'\n')

print('Ylos',Ylos.shape)
print(ndtotext(Ylos),'\n')

print('Zlos',Zlos.shape)
print(ndtotext(Zlos),'\n')

print('Zter',Zter.shape)
print(ndtotext(Zter),'\n')

print(ndtotext( Zter>Zlos ),'\n')

print(ndtotext(np.sum(Zter>Zlos,axis=0)>1),'\n')

print(ndtotext(V),'\n')
