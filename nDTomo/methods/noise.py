# -*- coding: utf-8 -*-
"""
Methods for adding artificial noise to data

@author: Antony Vamvakeros
"""

import numpy as np
from numpy.random import poisson
from tqdm import tqdm

def addpnoise1D(sp, ct):
    
    """
    Add Poisson noise to a 1D spectrum.

    Parameters
    ----------
    sp : ndarray
        1D array representing the spectrum (e.g., intensity values).
    ct : float
        Scaling factor representing the total counts or acquisition time.

    Returns
    -------
    ndarray
        Noisy spectrum with Poisson noise applied and rescaled by the count factor.
    """
    mi = min(sp)
    
    if mi < 0:
        
        sp = sp - sp + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        sp = sp + np.finfo(np.float32).eps
    
    return(poisson(sp * ct)/ ct)

def addpnoise2D(im, ct):
    
    """
    Add Poisson noise to a 2D image.

    Parameters
    ----------
    im : ndarray
        2D array representing the image (e.g., integrated intensity map).
    ct : float
        Scaling factor representing the total counts or acquisition time.

    Returns
    -------
    ndarray
        Noisy image with Poisson noise applied and rescaled by the count factor.
    """
    
    mi = min(im)
    
    if mi < 0:
        
        im = im - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        im = im + np.finfo(np.float32).eps
    
    return(poisson(im * ct)/ ct)

def addpnoise3D(vol, ct):
    '''
    Adds Poisson noise to a 3D hyperspectral volume (H x W x Bands),
    noise is added per pixel-spectrum (i.e., per (i,j,:)).
    
    Parameters
    ----------
    vol : ndarray
        3D hyperspectral image (H x W x Bands), must be non-negative.
    ct : float
        Scaling constant to simulate photon counts.
    '''
    vol = vol.copy()
    mi = np.min(vol)
    if mi < 0:
        vol = vol - mi + np.finfo(np.float32).eps
    elif mi == 0:
        vol = vol + np.finfo(np.float32).eps

    # Apply Poisson noise per pixel-spectrum
    noisy = np.random.poisson(vol * ct) / ct
    return noisy