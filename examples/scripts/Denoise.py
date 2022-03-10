# -*- coding: utf-8 -*-
"""
Denoising CT datasets

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import nDphantom_2D, load_example_patterns, nDphantom_3D, nDphantom_4D, nDphantom_2Dmap
from nDTomo.utils.misc import h5read_data, h5write_data, closefigs, showplot, showspectra, showim, showvol, normvol, addpnoise2D, addpnoise3D, cirmask, interpvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.ct.astra_tomo import astra_create_geo, astre_rec_vol, astre_rec_alg, astra_create_sino_geo, astra_create_sino, nDphantom_3D_sinograms, nDphantom_3D_FBP
from nDTomo.ct.conv_tomo import radonvol, fbpvol
from nDTomo.nn.models_tf import DCNN2D, DnCNN, DCNN1D
from nDTomo.nn.losses_tf import ssim_mae_loss, ssim_loss

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, h5py
from skimage.transform import rotate

import tensorflow as tf


#%% Ground truth

'''
These are the five ground truth componet spectra
'''

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
spectra = [dpAl, dpCu, dpFe, dpPt, dpZn]
showspectra([dpAl, dpCu + 0.1, dpFe + 0.2, dpPt + 0.3, dpZn + 0.4], 1)

'''
These are the five ground truth componet images
'''

npix = 200
# This creates a list containing five images, all with the same dimensions
iml = nDphantom_2D(npix, nim = 'Multiple')
print(len(iml))


imAl, imCu, imFe, imPt, imZn = iml

showim(imAl, 2)
showim(imCu, 3)
showim(imFe, 4)
showim(imPt, 5)
showim(imZn, 6)

#%% Ground truth data

gtimlist = [imAl, imCu, imFe, imPt, imZn]
gtsplist = [dpAl, dpCu, dpFe, dpPt, dpZn]
gtldlist = ['Al', 'Cu', 'Fe', 'Pt', 'Zn']


#%% Close the various figures

closefigs()

#%% Let's create a 3D (chemical-CT) dataset with two spatial and one spectral dimensions (x,y,spectral)

chemct = nDphantom_3D(npix, use_spectra = 'Yes', spectra = spectra, imgs = iml, indices = 'All',  norm = 'No')
print('The volume dimensions are %d, %d, %d' %(chemct.shape[0], chemct.shape[1], chemct.shape[2]))

#%% Normalise the volume

chemct = normvol(chemct)

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(chemct)
hs.explore()

#%% Let's perform a volume rendering

showvol(chemct)

#%% Create the sinograms

nproj = int(npix*np.pi/2/4)
chemsinos = nDphantom_3D_sinograms(chemct, nproj)

print('The sinogram dimensions are %d, %d, %d' %(chemsinos.shape[0], chemsinos.shape[1], chemsinos.shape[2]))

#%% Add noise to the sinograms

chemsinos = addpnoise3D(chemsinos, 50)

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(chemsinos)
hs.explore()

#%% Reconstruct the images

theta = np.deg2rad(np.arange(0, 180, 180/nproj))
chemct_vol1 = cirmask(nDphantom_3D_FBP(chemsinos[:,0::2,:], theta[0::2]))
chemct_vol2 = cirmask(nDphantom_3D_FBP(chemsinos[:,1::2,:], theta[1::2]))

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(np.concatenate((chemct_vol1, chemct_vol2), axis = 1))
hs.explore()

#%%

chemct_vol1 = np.transpose(chemct_vol1, (2,0,1))
chemct_vol2 = np.transpose(chemct_vol2, (2,0,1))
print(chemct_vol1.shape, chemct_vol2.shape)

#%% Normalise the images - doesn't help

for ii in tqdm(range(chemct_vol1.shape[0])):
    
    chemct_vol2[ii,:,:] = chemct_vol2[ii,:,:]/np.max(chemct_vol1[ii,:,:])
    chemct_vol1[ii,:,:] = chemct_vol1[ii,:,:]/np.max(chemct_vol1[ii,:,:])
    
#%%

train = np.concatenate((np.reshape(chemct_vol1, (chemct_vol1.shape[0], chemct_vol1.shape[1], chemct_vol1.shape[2], 1)),
                        np.reshape(chemct_vol2, (chemct_vol2.shape[0], chemct_vol2.shape[1], chemct_vol2.shape[2], 1))), axis = 0)

target = np.concatenate((np.reshape(chemct_vol2, (chemct_vol2.shape[0], chemct_vol2.shape[1], chemct_vol2.shape[2], 1)),
                        np.reshape(chemct_vol1, (chemct_vol1.shape[0], chemct_vol1.shape[1], chemct_vol1.shape[2], 1))), axis = 0)

print(train.shape, target.shape)

#% Mix the data

nim = train.shape[0]
index_vol = np.arange(nim)
np.random.shuffle(index_vol)

train = train[index_vol,:,:,:]
target = target[index_vol,:,:,:]

print(train.shape, target.shape)

#%% nDTomo DCNN2D

npix = chemct_vol1.shape[1]

model = DCNN2D(npix, nlayers=3, net='unet', dropout='No', batchnorm = 'No', filtnums=128, nconvs=3, actlayerfi = 'linear', skipcon = 'Yes')

model.summary()

#%% 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

#%%

my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=1E-5)]

model.fit(target[0:200,:,:,:], train[0:200,:,:,:],
    epochs = 200,
    verbose = True,
    batch_size = 1,
    callbacks=my_callbacks)

#%%

# Extract the history from the training object
history = model.history.history

plt.figure(1);plt.clf();
# Plot the training loss 
plt.plot(history['loss'][1:])


#%%

ii = np.random.randint(chemct_vol1.shape[0])

im = (chemct_vol1[ii,:,:] + chemct_vol2[ii,:,:])/2
im = tf.reshape(im, (1, im.shape[0], im.shape[1],1))
imro1 = model.predict(im)[0,:,:,0]

plt.figure(1);plt.clf()
plt.imshow(np.concatenate((im[0,:,:,0],imro1[:,:]), axis = 1), cmap = 'jet')
plt.colorbar()
plt.clim(0, np.max(imro1)+0.1)
# plt.axis('tight')
plt.title(ii)
plt.show()

#%%

vol = (chemct_vol1 + chemct_vol2)/2
vol = tf.reshape(vol, (vol.shape[0], vol.shape[1], vol.shape[2], 1))
chemctp = model.predict(vol, batch_size = 1)[:,:,:,0]

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(np.concatenate(((chemct_vol1 + chemct_vol2)/2, chemctp), axis = 2).transpose(1,2,0))
hs.explore()












