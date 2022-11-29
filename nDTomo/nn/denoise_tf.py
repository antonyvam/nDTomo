# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:14:21 2022

@author: Antony Vamvakeros
"""

import numpy as np
import time
from twdm import tqdm

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
