# -*- coding: utf-8 -*-
"""
Misc tools for nDTomo

@author: Antony
"""

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


















