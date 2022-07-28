# -*- coding: utf-8 -*-
"""
Clustering chemical imaging data using CNNs

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import nDphantom_2D, load_example_patterns, nDphantom_3D, nDphantom_4D, nDphantom_2Dmap
from nDTomo.utils.misc import h5read_data, h5write_data, closefigs, showplot, showspectra, showim, normvol, addpnoise2D, addpnoise3D, interpvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.misc3D import showvol
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.ct.astra_tomo import astra_create_geo, astra_rec_vol, astra_rec_alg, astra_create_sino_geo, astra_create_sino
from nDTomo.ct.conv_tomo import radonvol, fbpvol
from nDTomo.nn.models_tf import DCNN2D, DnCNN, Dense1D, Dense2D
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

npix = 512
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

#%% Random rotation - this is bad for clustering

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

model = DCNN2D(npix, nlayers=3, net='autoencoder', dropout='Yes', batchnorm = 'Yes', 
               filtnums=64, nconvs=3, actlayerfi = 'linear')

model.summary()

#%%

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')

my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=1E-5)]

model.fit(train, train,
    epochs = 100,
    validation_split=0.1,
    verbose = True,
    batch_size = 1,
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
print(ind)
imp = model.predict(train[ind:ind+1,:,:,:])[0,:,:,0]
imp = np.where(imp<0, 0, imp)
imc = np.concatenate((imp, train[ind,:,:,0]), axis = 1)

showim(imc, 1)

#%%

encoder = tf.keras.models.Model(model.input, model.layers[-21].output)
encoder.summary()

#%%

features = np.zeros((train.shape[0], 64))

for ii in tqdm(range(train.shape[0])):

    features[ii,:] = np.sum(np.sum(encoder.predict(train[ii:ii+1,:,:,:]),axis=3),axis=2)[0,:]


#%%

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
digit_features_tsne = tsne.fit_transform(features)

plt.scatter(digit_features_tsne[:,0], digit_features_tsne[:,1])

#%% use features for clustering

from sklearn.cluster import KMeans
clusters = 20
kmeans = KMeans(n_clusters=clusters, n_init=50)

pred = kmeans.fit_predict(features)

plt.figure(2);plt.clf();
plt.scatter(digit_features_tsne[:,0], digit_features_tsne[:,1], c=pred)

#%%

imagelist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    imagelist.append(np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2))


plotfigs_imgs(imagelist, rows=4, cols=5, figsize=(20,6), cl=True)




















