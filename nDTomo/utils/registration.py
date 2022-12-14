# -*- coding: utf-8 -*-
"""
Methods for registering data

@author: Antony Vamvakeros
"""

import matplotlib.pyplot as plt
from pystackreg import StackReg
from tqdm import tqdm
from numpy import concatenate

def regimage(ref, mov):

    '''
    Register an image using a reference image
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    reg = sr.register_transform(ref, mov)
    tmat = sr.register(ref, mov)
    reg = sr.transform(mov, tmat)
    
    plt.figure(1);plt.clf()
    plt.imshow(concatenate((ref, mov, reg), axis = 1), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()
    
    plt.figure(2);plt.clf()
    plt.imshow(concatenate((ref, mov, reg), axis = 0), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()
    
    plt.figure(3);plt.clf()
    plt.imshow(concatenate((mov - ref , reg - ref), axis = 1), cmap = 'jet')
    plt.clim(-0.3, 0.3)
    plt.colorbar()
    plt.show()

    return(reg, tmat)

def regvol(vol, tmat):
    
    '''
    Register a volume using a transformation matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    
    for ii in tqdm(range(vol.shape[2])):
        
        vol[:,:,ii] = sr.transform(vol[:,:,ii], tmat)
        
        print(ii)

    return(vol)

def regimtmat(im, tmat):

    '''
    Register an image using a transformation matrix
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    
    reg = sr.transform(im, tmat)
    
    return(reg)
