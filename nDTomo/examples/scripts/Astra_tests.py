# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:33:26 2020

@author: Antony
"""

import matplotlib.pyplot as plt
import time
from skimage.draw import random_shapes
# import tomopy
import numpy as np
import astra

#%%

def cirmask(im, npx=0):
    
    """
    
    Apply a circular mask to the image
    
    """
    
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2));
    
    dim =  im.shape
    if len(dim)==2:
        im = np.where(r>np.floor(sz/2) - npx,0,im)
    elif len(dim)==3:
        for ii in range(0,dim[2]):
            im[:,:,ii] = np.where(r>np.floor(sz/2),0,im[:,:,ii])
    return(im)

start=time.time()
# image, _ = random_shapes((128, 128), min_shapes=5, max_shapes=10,
                         # min_size=20, allow_overlap=True)
image, _ = random_shapes((50, 50), min_shapes=1, max_shapes=50, multichannel=False,
                         min_size=2, max_size=15, allow_overlap=True)
image = np.where(image==255, 0, image)
image = cirmask(image,5)
# image = np.random.poisson(image)
print((time.time()-start))

plt.figure(1);plt.clf();plt.imshow(image, cmap='jet');plt.show();

# #%%


# thetar = np.deg2rad(np.linspace(0,179,50))
# # phantom = np.transpose(np.array([image]),(2,1,0))
# phantom = image

# ang = tomopy.angles(nang=50, ang1=0, ang2=179)
# print(ang)

# print(thetar.shape, phantom.shape)

#%

phantom = image

# Create a basic square volume geometry
vol_geom = astra.create_vol_geom(phantom.shape[0], phantom.shape[0])
# Create a parallel beam geometry with 180 angles between 0 and pi, and
# 384 detector pixels of width 1.
# For more details on available geometries, see the online help of the
# function astra_create_proj_geom .
proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*phantom.shape[0]), np.linspace(0,np.pi,phantom.shape[0],False))
# Create a sinogram using the GPU.
# Note that the first time the GPU is accessed, there may be a delay
# of up to 10 seconds for initialization.
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

start=time.time()
sinogram_id, sinogram = astra.create_sino(phantom, proj_id)
print((time.time()-start))

sinogram = np.random.poisson(sinogram)

plt.figure(1);plt.clf();plt.imshow(sinogram, cmap='jet');plt.show();


#%

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
# cfg = astra.astra_dict('SIRT_CUDA')
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = { 'FilterType': 'shepp-logan' }

# Available algorithms:
# SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)
# possible values for FilterType:
# none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
# triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
# blackman-nuttall, flat-top, kaiser, parzen

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# astra.algorithm.run(alg_id, 150)
astra.algorithm.run(alg_id)

# Get the result
start=time.time()
rec = astra.data2d.get(rec_id)
print((time.time()-start))

rec = np.where(rec<0, 0, rec)
rec = cirmask(rec)

plt.figure(2);plt.clf();plt.imshow(np.concatenate((rec,image),axis=1), cmap='jet');plt.show();


astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
#%%
# prj = tomopy.project(obj=phantom, theta=thetar, pad=True)

#%%

import SampleGen as sg

sml = sg.random_sample()

sml.create_image(sz=50, misz=2, masz=15, mxshapes=50)

plt.figure(1);plt.clf();plt.imshow(sml.im, cmap='jet');plt.colorbar();plt.show();

sml.create_sino_geo()
sml.create_sino()

plt.figure(2);plt.clf();plt.imshow(sml.sinogram, cmap='jet');plt.colorbar();plt.show();
#%%

nsets = 10
nims = 100000

for ii in range(nsets):
    
    start=time.time()
    sml.create_imagestack(nims)
    print((time.time()-start))
    
    start=time.time()
    sml.create_sinostack()
    print((time.time()-start))
    
    fn = 'test_%d' %ii
    sml.export_imagestack('D:\\Dropbox (Finden)\\Finden team folder\\AI\\Parallax\\Libraries\\Images', fn)
    sml.export_sinostack('D:\\Dropbox (Finden)\\Finden team folder\\AI\\Parallax\\Libraries\\Sinograms', fn)

    # sml.astraclean()



plt.figure(1);plt.clf();plt.imshow(sml.vol[:,:,5], cmap='jet');plt.colorbar();plt.show();
plt.figure(2);plt.clf();plt.imshow(sml.sinograms[:,:,5], cmap='jet');plt.colorbar();plt.show();


#%%
# calculate mean value from RGB channels and flatten to 1D array
vals = sml.im.flatten()
vals = vals*255
# calculate histogram
counts, bins = np.histogram(vals, range(257))

phases = np.unique(counts)
# plot histogram centered on values 0..255
plt.figure(3);plt.clf();
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
# plt.xlim([-0.5, 255.5])
plt.show()



