# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:57:46 2019

@author: Antony
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import nDTomo.ct.paralleltomo


#%%


nT = 251
nP = 251
theta = np.arange(0, 180, 180/nP)

A, b, x = paralleltomo.paralleltomo(nT, theta, nT, nT)

#x = np.reshape(x, (nT,nT))
#x = np.transpose(x)
#x = np.ndarray.flatten(x)

y = A*x

yn = np.reshape( y, (nP, nT))


plt.figure(1);plt.clf();plt.imshow(np.transpose(yn), cmap = 'jet');plt.colorbar();plt.show();

#%% CGLS

K = 25;
# Initialization.
k = np.max(K)

n = A.shape[1]

# Prepare for CG iteration.
x = np.zeros((n,))

d = sparse.csr_matrix.dot(np.transpose(A),b)

r = b
normr2 = np.dot(np.transpose(d),d)
#normr2 = sparse.csr_matrix.dot(np.transpose(d),d)

plt.figure(1);plt.clf();
# Iterate.
ksave = 0;
for j in range(0,k):

#  Update x and r vectors.
  Ad = sparse.csr_matrix.dot(A,d)
  alpha = normr2/(np.dot(np.transpose(Ad),Ad))

  x  = x + d*alpha

  r  = r - Ad*alpha
  s  = sparse.csr_matrix.dot(np.transpose(A),r)
  
#  Update d vector.
  normr2_new =  np.dot(np.transpose(s),s)
  beta = normr2_new/normr2
  normr2 = normr2_new
  d = s + d*beta
  
#  Save, if wanted.
  j

  xn = x; xn = np.reshape(xn,(nT,nT)); 
  xn = np.where(xn<0, 0, xn)
  xn = xn/np.max(xn);
  
  plt.imshow(xn, cmap = 'jet');plt.title(j);
  plt.pause(0.5);
  

