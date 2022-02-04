# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:16:43 2022

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import nDphantom_2D, load_example_patterns, nDphantom_3D, nDphantom_4D, nDphantom_2Dmap
from nDTomo.utils.misc import h5read_data, h5write_data, closefigs, showplot, showspectra, showim, showvol, normvol, addpnoise2D, addpnoise3D, interpvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.ct.astra_tomo import astra_create_geo, astre_rec_vol, astre_rec_alg, astra_create_sino_geo, astra_create_sino
from nDTomo.ct.conv_tomo import radonvol, fbpvol
from nDTomo.nn.models_tf import DCNN2D, DnCNN
from nDTomo.nn.losses_tf import ssim_mae_loss, ssim_loss

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, h5py
from skimage.transform import rotate

import tensorflow as tf

#%%

'''
Part 1: Data generation
'''

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

'''
We will create a chemical tomography phantom using nDTomo
Here we create an XRD-CT dataset using 5 chemical components; this corresponds to five unique spectra (diffraction patterns in this case) and five unique images
This is a 3D matrix (array) with dimenion sizes (x, y, spectral): 200 x 200 x 250
So this corresponds to 250 images, each having 200 x 200 pixels
The goal is to perform dimensionality reduction/cluster analysis and extract these five images and/or spectra
The various methods can be applied either to the image domain by treating the volume as a stack of images (250 images, each having 200 x 200 pixels), 
or in the spectral domain (200 x 200 spectra with 250 points in each spectrum)
'''

chemct = nDphantom_3D(npix, use_spectra = 'Yes', spectra = spectra, imgs = iml, indices = 'All',  norm = 'No')

print('The volume dimensions are %d, %d, %d' %(chemct.shape[0], chemct.shape[1], chemct.shape[2]))

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(chemct)
hs.explore()

#%% Let's perform a volume rendering

showvol(chemct)

#%% Normalise the images

for ii in tqdm(range(chemct.shape[2])):
    
    chemct[:,:,ii]  = chemct[:,:,ii]/np.max(chemct[:,:,ii])

#%% Random rotation

for ii in tqdm(range(chemct.shape[2])):

    angle = np.random.rand(1)*360
    chemct[:,:,ii] = rotate(chemct[:,:,ii], angle[0])

#%% Mix the data

nim = chemct.shape[2]
index = np.arange(nim)
np.random.shuffle(index)

chemct = chemct[:,:,index]

#%%

train = tf.reshape(chemct, (chemct.shape[0], chemct.shape[1], chemct.shape[2], 1))
train = tf.transpose(train, (2,0,1,3))

print(train.shape)

#%% nDTomo DCNN2D

npix = chemct.shape[0]

model = DCNN2D(npix, nlayers=2, net='autoencoder', dropout='No', batchnorm = 'No', filtnums=128, nconvs=4, actlayerfi = 'relu')

model.summary()


#%% Dense Fully Connected Network

full_dim = chemct.shape[0]
# these are the downsampling/upsampling dimensions
encoding_dim1 = 100
encoding_dim2 = 100
encoding_dim3 = 100 # we will use these 3 dimensions for clustering

# This is our encoder input 
encoder_input_data = tf.keras.Input(shape=(full_dim,full_dim,1))
encoder_input_data_flat = tf.keras.layers.Flatten()(encoder_input_data)
# the encoded representation of the input
encoded_layer1 = tf.keras.layers.Dense(encoding_dim1, activation='relu')(encoder_input_data_flat)
encoded_layer1 = tf.keras.layers.Dropout(0.05)(encoded_layer1)
# encoded_layer1 = tf.keras.layers.BatchNormalization()(encoded_layer1)
encoded_layer2 = tf.keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer1)
encoded_layer2 = tf.keras.layers.Dropout(0.05)(encoded_layer2)
# encoded_layer2 = tf.keras.layers.BatchNormalization()(encoded_layer2)
# Note that encoded_layer3 is our 3 dimensional "clustered" layer, which we will later use for clustering
encoded_layer3 = tf.keras.layers.Dense(encoding_dim3, activation='relu', name="ClusteringLayer")(encoded_layer2)
encoded_layer3 = tf.keras.layers.Dropout(0.05)(encoded_layer3)
# encoded_layer1 = tf.keras.layers.BatchNormalization()(encoded_layer1)

# encoder_model = tf.keras.Model(encoder_input_data, encoded_layer3)

# the reconstruction of the input
decoded_layer3 = tf.keras.layers.Dense(encoding_dim2, activation='relu')(encoded_layer3)
decoded_layer3 = tf.keras.layers.Dropout(0.05)(decoded_layer3)
# decoded_layer3 = tf.keras.layers.BatchNormalization()(decoded_layer3)
decoded_layer2 = tf.keras.layers.Dense(encoding_dim1, activation='relu')(decoded_layer3)
decoded_layer2 = tf.keras.layers.Dropout(0.05)(decoded_layer2)
# decoded_layer2 = tf.keras.layers.BatchNormalization()(decoded_layer2)
decoded_layer1 = tf.keras.layers.Dense(full_dim * full_dim, activation='relu')(decoded_layer2)
decoded_layer1 = tf.keras.layers.Dropout(0.05)(decoded_layer1)
# decoded_layer1 = tf.keras.layers.BatchNormalization()(decoded_layer1)
decoded_layer1 = tf.keras.layers.Reshape((full_dim, full_dim, 1))(decoded_layer1)  

# This model maps an input to its autoencoder reconstruction
model = tf.keras.Model(encoder_input_data, outputs=decoded_layer1, name="Encoder")
model.summary()


#%%
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=ssim_loss)

my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=1E-5)]

model.fit(train, train,
    epochs = 500,
    validation_split=0.1,
    verbose = True,
    batch_size = 10,
    callbacks = my_callbacks)

#%%

# Extract the history from the training object
history = model.history.history

plt.figure(1);plt.clf();
# Plot the training loss 
plt.plot(history['loss'][1:])
plt.plot(history['val_loss'][1:])

#%%

ind = np.random.randint(0, 250)

im = model.predict(train[ind:ind+1,:,:,:])[0,:,:,0]

showim(train[ind,:,:,0], 1)
showim(im, 2)






















