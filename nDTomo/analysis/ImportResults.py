# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:40:16 2019

@author: Antony
"""

import h5py

def loadpeakfits(fn, peaks):
    '''
    It takes a filename (use .h5 or .hdf5) and the Areas, Pos, Sigma which are dictionaries
    '''

    Areas = {}
    Pos = {}
    Sigma = {}
    
    with h5py.File(fn,'r') as f:
        
        for ii in range(0,peaks):
            
            Areas[ii] = f['/Peak_Area_%d' %ii][:]
            Pos[ii] = f['/Peak_Position_%d' %ii][:]
            Sigma[ii] = f['/Peak_FWHM_%d' %ii][:]
            
        Bkga = f['/Bkga'][:]
        Bkgb = f['/Bkgb'][:]
            
        try:
            Bkgc = f['/Bkgc'][:]
        except:
            pass
        
    return Areas, Pos, Sigma, Bkga, Bkgb, Bkgc