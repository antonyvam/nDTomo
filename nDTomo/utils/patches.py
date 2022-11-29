# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:02:33 2022

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

