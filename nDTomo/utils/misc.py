# -*- coding: utf-8 -*-
"""
Misc tools for nDTomo

@author: Antony
"""

import numpy as np
import pkgutil

def ndtomopath():
    
    '''
    Finds the absolute path of the nDTomo software
    '''
    
    package = pkgutil.get_loader('nDTomo')
    ndtomo_path = package.path
    ndtomo_path = ndtomo_path.split('__init__.py')[0]
            
    return(ndtomo_path)

def addpnoise1D(sp, ct):
    
    mi = np.min(im)
    
    if sp < 0:
        
        sp = sp - sp + np.finfo(np.float32).eps
        
    elif sp == 0:
        
        sp = sp + np.finfo(np.float32).eps
    
    return(np.random.poisson(sp * ct)/ ct)

def addpnoise2D(im, ct):
    
    mi = np.min(im)
    
    if mi < 0:
        
        im = im - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        im = im + np.finfo(np.float32).eps
    
    return(np.random.poisson(im * ct)/ ct)

def addpnoise3D(vol, ct):
    
    '''
    Adds poisspn noise to a stack of images, 3rd dimension is z/spectral
    '''
    
    mi = np.min(vol)
    
    if vol < 0:
        
        vol = vol - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        vol = vol + np.finfo(np.float32).eps
        
    
    for ii in range(vol.shape[2]):
        
        vol[:,:,ii] = np.random.poisson(vol[:,:,ii] * ct)/ ct
    
    
    return(vol)


def cirmask(im, npx=0):
    
    """
    
    Apply a circular mask to the image/volume
    
    """
    
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2));
    
    dim =  im.shape
    if len(dim)==2:
        im = np.where(r>np.floor(sz/2) - npx,0,im)
    elif len(dim)==3:
        for ii in range(0,dim[2]):
            im[:,:,ii] = np.where(r>np.floor(sz/2),0,im[:,:,ii])
    return(im)















