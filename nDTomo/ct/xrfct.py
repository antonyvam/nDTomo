# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:01:36 2022

@author: Antony Vamvakeros
"""

from nDTomo.ct.conv_tomo import scalesinos, sinocentering
from nDTomo.ct.astra_tomo import astra_rec_single
from nDTomo.utils.misc import cirmask

import numpy as np
import os, h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"


def read_xrfct_data(home, sample, dataset):
    
    '''
    Reads xrf-ct data saved in h5 files
    Example:
    home = 'Y:\\2022\\ch1010\\'
    sample = 'sample1'
    dataset = '2'    
    '''
    
    fname = sample + '_' + dataset
    h5name = os.path.join(home, sample, fname, fname + '.h5')
    
    with h5py.File(h5name,'r') as hin:
        scans = list(hin['/'])
        print(list(hin[scans[0]]['measurement']))
    
    
    rot = read_data(h5name, 'measurement/rot_center' , scans )
    
    npx = [len(r) for r in rot]
    rot = np.concatenate( rot )
    ctr = np.concatenate( read_data(h5name, 'measurement/mca_det0' , scans ) )
    
    dty = read_data(h5name, 'instrument/positioners/dty', scans )
    dty = np.concatenate( [ np.full( n, y ) for n,y in zip(npx, dty)  ] )
    
    print(ctr.shape)
    return(ctr)

def read_data(h5name, counter, scans ):
    data = []
    with h5py.File(h5name,'r') as hin:
        for scan in tqdm(scans):
            if not scan.endswith('.1'):
                continue
            data.append( hin[scan][counter][()] )
    return data


def global_sino(ctr, nproj=1448, plot=True):
    
    ss = np.sum(ctr, axis=1)
    
    ss = np.reshape(ss, (int(ss.shape[0]/nproj), nproj))
    ss[0::2,:] = np.fliplr(ss[0::2,:])
    # ss = sinocentering(ss)
    ss = scalesinos(ss)
    
    if plot:
        plt.figure()
        plt.imshow(ss, cmap = 'jet')
        plt.axis('tight')
        plt.show()
    
    return(ss)


def global_im(ss, plot=True):

    r = astra_rec_single(ss)
    r[r<0] = 0
    r = cirmask(r, 5)
    
    if plot:    
        plt.figure()
        plt.imshow(r, cmap = 'jet')
        plt.colorbar()
        plt.axis('tight')
        plt.show()
    
    return(r)

def reshape_data(ctr, nproj):
    
    s = np.reshape(ctr, (int(ctr.shape[0]/nproj), nproj, ctr.shape[1]))
    s[0::2,:,:] = s[0::2,::-1,:]
    s = s/np.max(s)
    s = np.array(s, dtype='float32')
    return(s)




