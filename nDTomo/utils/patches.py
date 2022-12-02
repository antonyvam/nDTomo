# -*- coding: utf-8 -*-
"""
Methods for data splitting

@author: Antony Vamvakeros
"""

from nDTomo.utils.misc import even_idx, odd_idx, EMPatches
import numpy as np
from tqdm import tqdm

def create_2subvols(vol):
    
    voleven = np.zeros((int(vol.shape[0]/2), int(vol.shape[1]/2), vol.shape[2]))
    volodd = np.zeros((int(vol.shape[0]/2), int(vol.shape[1]/2), vol.shape[2]))
    
    for ii in tqdm(range(vol.shape[2])):

        voleven[:,:,ii] = even_idx(vol[:,:,ii])
        volodd[:,:,ii] = odd_idx(vol[:,:,ii])

    return(voleven, volodd)


def create_2vols1D(vol):
    
    voleven, volodd = create_2subvols(vol)
    
    voleven = np.reshape(voleven, (voleven.shape[0]*voleven.shape[1], voleven.shape[2], 1))
    volodd = np.reshape(volodd, (volodd.shape[0]*volodd.shape[1], volodd.shape[2], 1))
    
    train = np.concatenate((voleven, volodd), axis = 0)
    target = np.concatenate((volodd, voleven), axis = 0)

    # Mix
    
    inds = np.arange(train.shape[0])
    np.random.shuffle(inds)
    
    train = np.float32(train[inds,:,:])
    target = np.float32(target[inds,:,:])
    
    print(train.shape, target.shape)
    
    return(train, target)


def create_left_up_ims(im):

    '''
    Create four images by taking the odd/even pixels and squeezing up/left
    This is the method used in the Noise2Fast method
    '''    

    im_even_left = np.zeros((im.shape[0], int(im.shape[1]/2)), dtype = 'float32')
    im_even_left[0::2,:] = im[0::2,0::2]
    im_even_left[1::2,:] = im[1::2,1::2]
    
    im_odd_left = np.zeros((im.shape[0], int(im.shape[1]/2)), dtype = 'float32')
    im_odd_left[0::2,:] = im[0::2,1::2]
    im_odd_left[1::2,:] = im[1::2,0::2]
        
    im_even_up = np.zeros((int(im.shape[0]/2), im.shape[1]), dtype = 'float32')
    im_even_up[:,0::2] = im[0::2,0::2]
    im_even_up[:,1::2] = im[1::2,1::2]
    
    im_odd_up = np.zeros((int(im.shape[0]/2), im.shape[1]), dtype = 'float32')
    im_odd_up[:,0::2] = im[1::2,0::2]
    im_odd_up[:,1::2] = im[0::2,1::2]
        
    return(im_even_left, im_odd_left, im_even_up, im_odd_up)


def create_odd_even_ims(im):
    
    '''
    Create four images by taking the odd/even pixels without squeezing the pixels
    '''
    
    im00 = im[0::2,0::2]
    im01 = im[0::2,1::2]
    im11 = im[1::2,1::2]
    im10 = im[1::2,0::2]
    
    return(im00, im01, im11, im10)


def create_im_patches(im_list, patchsize = 32, overlap = 0):

    emp = EMPatches()
    
    patches = []
    
    for im in range(len(im_list)):

        im_patches, indices = emp.extract_patches(im_list[im], patchsize=patchsize, overlap=overlap)
        patches.append(np.array(im_patches, dtype='float32'))

    return(patches)

















