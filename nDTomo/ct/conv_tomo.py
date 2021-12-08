# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony
"""


import numpy as np
from skimage.transform import iradon, radon



def radonvol(vol, nproj, scan = 180):
    
    '''
    Calculates the radon transform of a stack of images, 3rd dimension is z/spectral
    '''
    
    theta = np.arange(0, scan, scan/nproj)
    
    s = np.zeros((vol.shape[0], nproj, vol.shape[2]))
    
    for ii in range(s.shape[2]):
        
        s[:,:,ii] = radon(xrdct[:,:,ii], theta)
    
        print(ii)           
        
    
    print('The dimensions of the sinogram volume are ', s.shape)
        
    return(s)
        

def fbpvol(svol, scan = 180):
    
    '''
    Calculates the reconstructed images of a stack of sinograms using the filtered backprojection algorithm, 3rd dimension is z/spectral
    '''
    
    nt = svol.shape[0]
    nproj = svol.shape[1]
    
    theta = np.arange(0, scan, scan/nproj)
    
    vol = np.zeros((nt, nt, svol.shape[2]))
    
    for ii in range(svol.shape[2]):
        
        vol[:,:,ii] = iradon(svol[:,:,ii], theta, nt)
    
        print(ii)           
        
    
    print('The dimensions of the reconstructed volume are ', vol.shape)
        
    return(vol)
    
    
    