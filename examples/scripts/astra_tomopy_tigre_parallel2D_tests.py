# -*- coding: utf-8 -*-
"""
Reconstruction tests using the astra toolbox, tomopy and tigre

@authors: Kenan Gnonhoue and Antony Vamvakeros
"""

#%% First let's import some necessary modules

from datetime import date
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.sim.shapes.phantoms import SheppLogan
from nDTomo.utils.misc import closefigs, addpnoise2D, addpnoise3D, interpvol, showplot, showspectra, showim, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.misc3D import showvol
from nDTomo.ct.astra_tomo import astra_create_sino, astra_rec_single
from nDTomo.utils.misc import cirmask

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, h5py

%matplotlib auto

#%% Create a Shepp-Logan phantom

npix = 256
im = SheppLogan(npix)

showim(im, 1, cmap='jet')

#%% Create sinogram

sino = astra_create_sino(im)

showim(sino, 2, cmap='jet')

rec = astra_rec_single(sino.transpose())

showim(rec, 3, cmap='jet')

#%% Noisy sinogram

sino_noisy = addpnoise2D(sino, ct=10)

showim(sino_noisy, 2, cmap='jet')

rec = astra_rec_single(sino_noisy.transpose())

showim(rec, 3, cmap='jet')

#%% Sinogram with angular undersampling

sino_under = astra_create_sino(im, npr=256/4)

showim(sino_under, 2, cmap='jet')

rec = astra_rec_single(sino_under.transpose())

showim(rec, 3, cmap='jet')

#%% Noisy sinogram with angular undersampling

sino_under = astra_create_sino(im, npr=256/4)
sino_under_noisy = addpnoise2D(sino_under, ct=10)

showim(sino_under_noisy, 2, cmap='jet')

rec = astra_rec_single(sino_under_noisy.transpose())

showim(rec, 3, cmap='jet')


#%% Let's try some other methods than FBP for the sinogram that has angular undersampling!


rec = astra_rec_single(sino_under.transpose(), method='SIRT_CUDA', nits=200)
# rec = astra_rec_single(sino_under.transpose(), method='SART_CUDA', nits=200)
# rec = astra_rec_single(sino_under.transpose(), method='ART', nits=100000)
# rec = astra_rec_single(sino_under.transpose(), method='CGLS', nits=20)

showim(rec, 1, cmap='jet')


#%% Let's try some other methods than FBP for the noisy sinogram!


rec = astra_rec_single(sino_noisy.transpose(), method='SIRT_CUDA', nits=200)
# rec = astra_rec_single(sino_noisy.transpose(), method='SART_CUDA', nits=200)
# rec = astra_rec_single(sino_noisy.transpose(), method='ART', nits=100000)
# rec = astra_rec_single(sino_noisy.transpose(), method='CGLS', nits=20)

showim(rec, 1, cmap='jet')


#%% Let's try tomopy

'''
The methods: art, bart, fbp, gridrec, mlem, osem, ospml_hybrid, ospml_quad, pml_hybrid, pml_quad, sirt, tv, grad, tikh
'''

import tomopy.recon as tomopyrec

npr = sino.shape[0]
theta = np.deg2rad(np.arange(0, 180, 180/npr))

start=time.time()

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='fbp', filter_name='ramlak')[0,:,:]

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
                # algorithm='art', num_iter=50)[0,:,:]

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='bart', num_iter=50)[0,:,:]

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='gridrec')[0,:,:]
# rec = np.where(rec<0, 0, rec)
# rec = cirmask(rec)

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='mlem', num_iter=50)[0,:,:]

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='osem', num_iter=50)[0,:,:]

# rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
#                 algorithm='sirt', num_iter=50)[0,:,:]

rec = tomopyrec(sino.reshape(sino.shape[0], 1, sino.shape[1]), theta, 
                algorithm='tv', num_iter=50)[0,:,:]



print((time.time()-start))

showim(rec, 1, cmap='jet')


#%% Let's try tigre!

'''
Details about the algorithms and their inputs can be found here: https://github.com/CERN/TIGRE/tree/master/Python/tigre/algorithms
'''

import tigre
import tigre.algorithms as algs


s = np.copy(sino_under)

npr = s.shape[0]
theta = np.deg2rad(np.arange(0, 180, 180/npr))

geo = tigre.geometry()

# Distances
geo.DSD = 1536  # Distance Source Detector      (mm)
geo.DSO = 1000  # Distance Source Origin        (mm)

geo.nVoxel = np.array([1, s.shape[1], s.shape[1],])  # number of voxels              (vx)
geo.sVoxel = np.array([1, s.shape[1], s.shape[1],])  # total size of the image       (mm)
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)
# Detector parameters
geo.nDetector = np.array([1, s.shape[1]])  # number of pixels              (px)
geo.dDetector = np.array([geo.dVoxel[0], 1])  # size of each pixel            (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
# Offsets
geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
# MAKE SURE THAT THE DETECTOR PIXELS SIZE IN V IS THE SAME AS THE IMAGE!

geo.mode = "parallel"

rec = algs.fbp(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta)[0,:,:].transpose()
# rec = algs.cgls(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta, 40)[0,:,:].transpose()
# rec = algs.mlem(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta, 100)[0,:,:].transpose()
# rec = algs.sirt(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta, 100)[0,:,:].transpose()
# rec = algs.sart(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta, 10)[0,:,:].transpose()
# rec = algs.fista(s.reshape(s.shape[0], 1, s.shape[1]), geo, theta, 1000)[0,:,:].transpose()

showim(rec, 1, cmap='jet')


#%% Plot and save an image

plt.figure(1);plt.clf()
plt.imshow(rec, cmap='jet')
plt.colorbar()
plt.show()
fn = 'C:\\Dropbox (Finden)\\Finden_Research\\Active_Projects\\Fuse\\results\\sino_under_tigre_FBP.png'
plt.savefig(fn)

#%% Export the results

fn = 'C:\\Dropbox (Finden)\\Finden_Research\\Active_Projects\\Fuse\\results\\sino_under_tigre_FBP.h5'

with h5py.File(fn, 'w') as f:

    f.create_dataset('sinogram', s)
    f.create_dataset('rec', rec)
    f.create_dataset('method', 'FBP')
    f.close()


#%% Calculate some metrics using tensorflow

import tensorflow as tf

im = np.array(im, dtype='float64')
rec = np.array(rec, dtype='float64')

mae = tf.reduce_mean(tf.keras.losses.MAE(im, rec)).numpy()
mse = tf.reduce_mean(tf.keras.losses.MSE(im, rec)).numpy()
psnr = tf.image.psnr(im.reshape(1, im.shape[0],im.shape[0],1), rec.reshape(1, im.shape[0],im.shape[0],1), 1).numpy()
ssim = tf.image.ssim(im.reshape(1, im.shape[0],im.shape[0],1), rec.reshape(1, im.shape[0],im.shape[0],1), 1).numpy()

print(mae, mse, psnr, ssim)














