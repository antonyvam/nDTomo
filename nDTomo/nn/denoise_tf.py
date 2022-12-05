# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:14:21 2022

@author: Antony Vamvakeros
"""

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def denoise_vol1D(vol, model, gpu=True):

    if gpu:
        rn = np.copy(vol)
        noisy = np.reshape(rn, (rn.shape[0]*rn.shape[1], rn.shape[2]))
        noisy = np.float32(100*noisy)
        
        start = time.time()
        pred1d = model.predict(noisy, batch_size = 128, verbose=1)[:,:,0]
        print(time.time() - start)
        
        pred1d = np.reshape(pred1d, (rn.shape[0], rn.shape[1], noisy.shape[1]))
        
        noisy = np.copy(vol)
        noisy = np.float32(100*noisy)

    else:
        rn = np.copy(vol)
        noisy = np.reshape(rn, (rn.shape[0]*rn.shape[1], rn.shape[2]))
        noisy = np.float32(100*noisy)
        
        pred1d = np.zeros_like(noisy)
        kk = 0
        for ii in tqdm(np.arange(0,pred1d.shape[0], 128)):
            pred1d[kk*128:(kk+1)*128,:] = model.predict(noisy[kk*128:(kk+1)*128,:], verbose=0)[:,:,0]
            kk = kk + 1
        
        print(pred1d.shape)
        
        pred1d = np.reshape(pred1d, (rn.shape[0], rn.shape[1], noisy.shape[1]))
        
        noisy = np.copy(vol)
        noisy = np.float32(100*noisy)

    return(pred1d)


def create_global_mask(vol, thr = 0.1):
    
    '''
    Method to extract the indices from a region of interest from a 3D array
    Use a list containing the 3d arrays
    '''
    
    newvol = np.zeros_like(vol[0], dtype='float32')
    for ii in range(len(vol)):
        newvol = newvol + vol[ii]
    
    mask = np.sum(newvol,axis=2)
    mask = mask/np.max(mask)
    mask = np.where(mask<0.1, 0, 1)
    
    plt.figure()
    plt.imshow(mask, cmap = 'gray');
    plt.colorbar()
    plt.show()
    
    inds = np.where(mask>0)    
    
    return(inds)
    
    
def roisp(vol, inds, nsp=10000):
    
    rows = inds[0]
    cols = inds[1]
    nch = vol.shape[2]
    
    inds = np.arange(len(rows))
    np.random.shuffle(inds)
    inds = inds[:nsp]
        
    vol = np.reshape(vol[rows[inds], cols[inds],:], (nsp, nch, 1))    
    
    return(vol)