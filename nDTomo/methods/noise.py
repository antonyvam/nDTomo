# -*- coding: utf-8 -*-
"""
Methods for adding artificial noise to data

@author: Antony Vamvakeros
"""

from numpy import min, finfo, float32
from numpy.random import poisson
from tqdm import tqdm

def addpnoise1D(sp, ct):
    
    '''
    Adds poisson noise to a spectrum
    '''    
    mi = min(sp)
    
    if mi < 0:
        
        sp = sp - sp + finfo(float32).eps
        
    elif mi == 0:
        
        sp = sp + finfo(float32).eps
    
    return(poisson(sp * ct)/ ct)

def addpnoise2D(im, ct):
    
    '''
    Adds poisson noise to an image
    '''
    
    mi = min(im)
    
    if mi < 0:
        
        im = im - mi + finfo(float32).eps
        
    elif mi == 0:
        
        im = im + finfo(float32).eps
    
    return(poisson(im * ct)/ ct)

def addpnoise3D(vol, ct):
    
    '''
    Adds poisson noise to a stack of images, 3rd dimension is z/spectral
    '''
    
    mi = min(vol)
    
    if mi < 0:
        
        vol = vol - mi + finfo(float32).eps
        
    elif mi == 0:
        
        vol = vol + finfo(float32).eps
        
    
    for ii in tqdm(range(vol.shape[2])):
        
        vol[:,:,ii] = poisson(vol[:,:,ii] * ct)/ ct
    
    
    return(vol)