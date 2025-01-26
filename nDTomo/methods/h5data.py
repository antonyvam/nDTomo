# -*- coding: utf-8 -*-
"""
Methods for X-rays

@author: Antony Vamvakeros
"""

import h5py
from numpy import array

def h5read(filename):
    df = {}
    with h5py.File(filename, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())
    
        # Get the data
        for key in a_group_key:
            
            df[key] = array(list(f[key][()]))
            
    return(df)

def h5read_keys(fn):

    '''
    Reads a dataset from an h5 file.
    Inputs:
        fn: full path to h5 file, e.g. 'C:\\path_to_data\\file.h5'
    '''
        
    with h5py.File(fn, "r") as f:
    
        datasets = f.keys() 
        print(datasets)

    return(datasets)

def h5write_dataset(p, fn, data):

    '''
    Write a dataset to an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        data: dataset e.g. numpy array
    '''
    
    f = "%s%s.h5" %(p,fn)
    
    h5f = h5py.File(f, "w")
    
    h5f.create_dataset('data', data = data)
    
    h5f.close()


def h5read_dataset(fn, dataset = 'data'):

    '''
    Reads a dataset from an h5 file.
    Inputs:
        fn: full path to h5 file, e.g. 'C:\\path_to_data\\file.h5'
        data: string corresponding to the h5 dataset
    '''
        
    with h5py.File(fn, "r") as h5f:
    
        data = []    
    
        try:
            
            data = h5f[dataset][:]
            
        except:
            
            print('/%s is not available in the h5 file' %dataset)

    return(data)

def h5write_data(p, fn, dlist, data):

    '''
    Write a dataset to an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        dlist: list containing the names for the various datasets
        data:list containing the various dataset e.g. list of numpy arrays
    '''
        
    f = "%s%s.h5" %(p,fn)
    
    h5f = h5py.File(f, "w")
    
    for ii in range(len(dlist)):
    
        h5f.create_dataset(dlist[ii], data = data[ii])
    
    h5f.close()



def h5read_data(fn, dlist):

    '''
    Reads a dataset from an h5 file.
    Inputs:
        fn: full path to h5 file, e.g. 'C:\\path_to_data\\file.h5'
        dlist: list containing the names for the various datasets
    Outputs:
        data: dictionary containing the various datasets
    '''
        
    with h5py.File(fn, "r") as h5f:
    
        data = {}
        
        for ii in range(len(dlist)):

            try:
                
                data[dlist[ii]] = h5f[dlist[ii]][:]
                
            except:
                
                print('/%s is not available in the h5 file' %dlist[ii])

    return(data)