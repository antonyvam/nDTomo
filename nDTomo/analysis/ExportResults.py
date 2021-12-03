# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:55:36 2019

@author: Antony
"""
import h5py

def savepeakfits(fn, Areas, Pos, Sigma):
    '''
    It takes a filename (use .h5 or .hdf5) and the Areas, Pos, Sigma which are dictionaries
    '''
    
    h5f = h5py.File(fn, "w")
    
    for ii in range(0,len(Areas.keys())):
        
        h5f.create_dataset('Peak_Area_%d' %ii, data=Areas[ii])
        h5f.create_dataset('Peak_Position_%d' %ii, data=Pos[ii])
        h5f.create_dataset('Peak_FWHM_%d' %ii, data=Sigma[ii])
        
    h5f.close()