# -*- coding: utf-8 -*-
"""
Methods for data splitting

@author: Antony Vamvakeros
"""

from nDTomo.utils.misc import even_idx, odd_idx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    inds = []
    
    for im in range(len(im_list)):

        im_patches, indices = emp.extract_patches(im_list[im], patchsize=patchsize, overlap=overlap)
        patches.append(np.array(im_patches, dtype='float32'))
        inds.append(indices)

    patches = np.array(patches, dtype='float32')

    return(patches, inds)

def create_global_mask(vol, thr = 0.1):
    
    '''
    Method to extract the indices from a region of interest from a 3D array
    Use a list containing the 3d arrays
    '''
    
    newvol = np.zeros_like(vol[0], dtype='float32')
    for ii in range(len(vol)):
        newvol = newvol + vol[ii]
    
    mask = np.sum(newvol,axis=2)
    mask = mask/np.max(mask)
    mask = np.where(mask<0.1, 0, 1)
    
    plt.figure()
    plt.imshow(mask, cmap = 'gray');
    plt.colorbar()
    plt.show()
    
    inds = np.where(mask>0)    
    
    return(inds)


def roisp(vol, inds, nsp=10000):
    
    rows = inds[0]
    cols = inds[1]
    nch = vol.shape[2]
    
    inds = np.arange(len(rows))
    np.random.shuffle(inds)
    inds = inds[:nsp]
        
    vol = np.reshape(vol[rows[inds], cols[inds],:], (nsp, nch, 1))    
    
    return(vol)









class EMPatches(object):
    
    '''
    Original code: https://github.com/Mr-TalhaIlyas/EMPatches
    AV added patching for 1D and 3D data
    '''
    
    def __init__(self):
        pass

    def extract_patches(self, data, patchsize, overlap=0, stride=None, datatype = 'Image'):
        '''
        Parameters
        ----------
        data : array to extract patches from; it can be 1D, 2D or 3D [W, H, D]. H: Height, W: Width, D: Depth
        patchsize :  size of patch to extract from image only square patches can be
                     extracted for now.
        overlap (Optional): overlap between patched in percentage a float between [0, 1].
        stride (Optional): Step size between patches
        datatype (Optional): data type, can be 'Spectrum', 'Image', 'Volume'
        Returns
        -------
        data_patches : a list containing extracted patches of images.
        indices : a list containing indices of patches in order, whihc can be used 
                  at later stage for 'merging_patches'.
    
        '''

        dims = data.shape

        if len(dims)==1:        

            width = data.shape[0]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            
        elif len(dims)==2: 

            height = data.shape[0]
            width = data.shape[1]
            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)

        elif len(dims)==3:
            
            height = data.shape[0]
            width = data.shape[1]
            depth = data.shape[2]

            maxWindowSize = patchsize
            windowSizeX = maxWindowSize
            windowSizeY = maxWindowSize
            windowSizeZ = maxWindowSize

            windowSizeX = min(windowSizeX, width)
            windowSizeY = min(windowSizeY, height)
            windowSizeZ = min(windowSizeZ, depth)
            

        if stride is not None:
            if len(dims)==1:
                stepSizeX = stride
            elif len(dims)==2:
                stepSizeX = stride
                stepSizeY = stride
            elif len(dims)==3:
                stepSizeX = stride
                stepSizeY = stride
                stepSizeZ = stride
                        
        elif overlap is not None:
            overlapPercent = overlap

            if len(dims)==1:
                windowSizeX = maxWindowSize     

                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
    
                # Compute the window overlap and step size
                windowOverlapX = int(np.floor(windowSizeX * overlapPercent))
    
                stepSizeX = windowSizeX - windowOverlapX
                
            elif len(dims)==2:
                windowSizeX = maxWindowSize
                windowSizeY = maxWindowSize

                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                windowSizeY = min(windowSizeY, height)
    
                # Compute the window overlap and step size
                windowOverlapX = int(np.floor(windowSizeX * overlapPercent))
                windowOverlapY = int(np.floor(windowSizeY * overlapPercent))
    
                stepSizeX = windowSizeX - windowOverlapX
                stepSizeY = windowSizeY - windowOverlapY
                
            elif len(dims)==3:
                windowSizeX = maxWindowSize
                windowSizeY = maxWindowSize
                windowSizeZ = maxWindowSize
                
                # If the input data is smaller than the specified window size,
                # clip the window size to the input size on both dimensions
                windowSizeX = min(windowSizeX, width)
                windowSizeY = min(windowSizeY, height)
                windowSizeZ = min(windowSizeZ, depth)
    
                # Compute the window overlap and step size
                windowOverlapX = int(np.floor(windowSizeX * overlapPercent))
                windowOverlapY = int(np.floor(windowSizeY * overlapPercent))
                windowOverlapZ = int(np.floor(windowSizeZ * overlapPercent))
    
                stepSizeX = windowSizeX - windowOverlapX
                stepSizeY = windowSizeY - windowOverlapY                
                stepSizeZ = windowSizeZ - windowOverlapZ                
        
        
        if len(dims)==1:

            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            xOffsets = list(range(0, lastX+1, stepSizeX))
            
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            	xOffsets.append(lastX)
            
            data_patches = []
            indices = []
            
            for xOffset in xOffsets:
                  if len(data.shape) >= 3:
                      data_patches.append(data[(slice(xOffset, xOffset+windowSizeX, None))])
                  else:
                      data_patches.append(data[(slice(xOffset, xOffset+windowSizeX))])
                      
                  indices.append((xOffset, xOffset+windowSizeX))
                  
        elif len(dims)==2:
    
            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            lastY = height - windowSizeY
            xOffsets = list(range(0, lastX+1, stepSizeX))
            yOffsets = list(range(0, lastY+1, stepSizeY))
            
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            	xOffsets.append(lastX)
            if len(yOffsets) == 0 or yOffsets[-1] != lastY:
            	yOffsets.append(lastY)
            
            data_patches = []
            indices = []
            
            for xOffset in xOffsets:
                for yOffset in yOffsets:
                  if len(data.shape) >= 3:
                      data_patches.append(data[(slice(yOffset, yOffset+windowSizeY, None),
                                              slice(xOffset, xOffset+windowSizeX, None))])
                  else:
                      data_patches.append(data[(slice(yOffset, yOffset+windowSizeY),
                                              slice(xOffset, xOffset+windowSizeX))])
                      
                  indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX))            
         
        elif len(dims)==3:
            
            # Determine how many windows we will need in order to cover the input data
            lastX = width - windowSizeX
            lastY = height - windowSizeY
            lastZ = depth - windowSizeZ
            
            xOffsets = list(range(0, lastX+1, stepSizeX))
            yOffsets = list(range(0, lastY+1, stepSizeY))
            zOffsets = list(range(0, lastZ+1, stepSizeZ))
            
            # Unless the input data dimensions are exact multiples of the step size,
            # we will need one additional row and column of windows to get 100% coverage
            if len(xOffsets) == 0 or xOffsets[-1] != lastX:
            	xOffsets.append(lastX)
            if len(yOffsets) == 0 or yOffsets[-1] != lastY:
            	yOffsets.append(lastY)
            if len(zOffsets) == 0 or zOffsets[-1] != lastZ:
            	zOffsets.append(lastZ)
            
            data_patches = []
            indices = []
            
            for xOffset in xOffsets:
                for yOffset in yOffsets:
                    for zOffset in zOffsets:
                      if len(data.shape) >= 4:
                          data_patches.append(data[(slice(yOffset, yOffset+windowSizeY, None),
                                                  slice(xOffset, xOffset+windowSizeX, None),
                                                  slice(zOffset, zOffset+windowSizeZ, None))])
                      else:
                          data_patches.append(data[(slice(yOffset, yOffset+windowSizeY),
                                                  slice(xOffset, xOffset+windowSizeX),
                                                  slice(zOffset, zOffset+windowSizeZ))])
                          
                      indices.append((yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX, zOffset, zOffset+windowSizeZ))   
        
        return data_patches, indices
    
    
    def merge_patches(self, data_patches, indices, mode='overwrite'):
        '''
        Parameters
        ----------
        data_patches : list containing image patches that needs to be joined, dtype=uint8
        indices : a list of indices generated by 'extract_patches' function of the format;
                    (yOffset, yOffset+windowSizeY, xOffset, xOffset+windowSizeX)
        mode : how to deal with overlapping patches;
                overwrite -> next patch will overwrite the overlapping area of the previous patch.
                max -> maximum value of overlapping area at each pixel will be written.
                min -> minimum value of overlapping area at each pixel will be written.
                avg -> mean/average value of overlapping area at each pixel will be written.
        Returns
        -------
        Stitched image.
        '''
        modes = ["overwrite", "max", "min", "avg"]
        if mode not in modes:
            raise ValueError(f"mode has to be either one of {modes}, but got {mode}")

        dims = len(indices[-1])
        
        if dims==2:
            orig_h = indices[-1][1]
        elif dims==4:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
        elif dims==6:
            orig_h = indices[-1][1]
            orig_w = indices[-1][3]
            orig_d = indices[-1][5]
        
        ### There is scope here for rgb/hyperspectral volume (i.e. 4D -> 3 spatial and 1 spectral dimensions, simplest case is only 3 channles for the spectral dimension)
        rgb = True
        if len(data_patches[0].shape) == 2:
            rgb = False
        
        if mode == 'min':
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) + np.inf # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, 3)).astype(np.float32) + np.inf # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) + np.inf # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) + np.inf # using float here is better
                
        else:
            if dims == 2:
                empty_data = np.zeros((orig_h)).astype(np.float32) # using float here is better
                
            elif dims==4:
                if rgb:
                    empty_data = np.zeros((orig_h, orig_w, 3)).astype(np.float32) # using float here is better
                else:
                    empty_data = np.zeros((orig_h, orig_w)).astype(np.float32) # using float here is better

            elif dims==6:
                empty_data = np.zeros((orig_h, orig_w, orig_d)).astype(np.float32) # using float here is better

        for i, indice in enumerate(indices):

            if mode == 'overwrite':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = data_patches[i]

                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = data_patches[i]
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = data_patches[i]
                        
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = data_patches[i]
                        
                        
            elif mode == 'max':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.maximum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])


            elif mode == 'min':
                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1]])
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], :])
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3]])
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.minimum(data_patches[i], empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])
                    
            elif mode == 'avg':

                if dims == 2:
                    empty_data[indice[0]:indice[1]] = np.where(empty_data[indice[0]:indice[1]] == 0,
                                                                                    data_patches[i], 
                                                                                    np.add(data_patches[i],empty_data[indice[0]:indice[1]])/2)
                elif dims == 4:
                    if rgb:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], :] == 0,
                                                                                            data_patches[i], 
                                                                                            np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], :])/2)
                        # Below line should work with np.ones mask but giving Weights sum to zero error and is approx. 10 times slower then np.where
                        # empty_data[indice[0]:indice[1], indice[2]:indice[3], :] = np.average(([empty_data[indice[0]:indice[1], indice[2]:indice[3], :],
                        #                                                                         data_patches[i]]), axis=0,
                        #                                                                         weights=(np.asarray([empty_data[indice[0]:indice[1], indice[2]:indice[3], :],
                        #                                                                                               data_patches[i]])>0))
                    else:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3]])/2)
                elif dims==6:
                        empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] = np.where(empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]] == 0,
                                                                                        data_patches[i], 
                                                                                        np.add(data_patches[i],empty_data[indice[0]:indice[1], indice[2]:indice[3], indice[4]:indice[5]])/2)

        return empty_data



        

def patch_via_indices(data, indices):
    '''
        Parameters
        ----------
        data : data of shape HxWxC or HxW.
        indices :   list of indices/tuple of 4 e.g;
                    [(ystart, yend, xstart, xend), -> indices of 1st patch
                     (ystart, yend, xstart, xend), -> indices of 2nd patch
                     ...]
        Returns
        -------
        data_patches : a list containing extracted patches of data.
        '''
    dims = len(indices[-1])

    data_patches=[]
    
    if dims==2:
    
        for indice in indices:
            data_patches.append(data[(slice(indice[0], indice[1]))])
        
    elif dims==4:
        
        for indice in indices:
            if len(data.shape) >= 3:
                data_patches.append(data[(slice(indice[0], indice[1], None),
                                        slice(indice[2], indice[3], None))])
            else:
                data_patches.append(data[(slice(indice[0], indice[1]),
                                        slice(indice[2], indice[3]))])
            
    elif dims==6:            
            
        for indice in indices:
            
            data_patches.append(data[(slice(indice[0], indice[1]),
                                    slice(indice[2], indice[3]),
                                    slice(indice[3], indice[4]))])            






