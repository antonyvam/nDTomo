# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import phantom5c_xanesct, phantom5c_xrdct, load_example_patterns, phantom5c, phantom5c_xrdct_images
from nDTomo.utils import hyperexpl
from nDTomo.utils.misc import addpnoise2D, addpnoise3D
import numpy as np
import matplotlib.pyplot as plt
import time, h5py

#%% Ground truth

'''
These are the five ground truth componet spectra
'''

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()

plt.figure(1);plt.clf()
plt.plot(dpAl)
plt.plot(dpCu + 0.1)
plt.plot(dpFe + 0.2)
plt.plot(dpPt + 0.3)
plt.plot(dpZn + 0.4)
plt.show()


'''
These are the five ground truth componet images
'''

npix = 200
imAl, imCu, imFe, imPt, imZn = phantom5c(npix)

plt.figure(2);plt.clf()
plt.imshow(imAl)
plt.show()

plt.figure(3);plt.clf()
plt.imshow(imCu)
plt.show()

plt.figure(4);plt.clf()
plt.imshow(imFe)
plt.show()

plt.figure(5);plt.clf()
plt.imshow(imPt)
plt.show()

plt.figure(6);plt.clf()
plt.imshow(imZn)
plt.show()


#%% Create the chemical tomography dataset

'''
We will create a chemical tomography phantom using nDTomo
Here we create an XRD-CT dataset using 5 chemical components; this corresponds to five unique spectra (diffraction patterns in this case) and five unique images
This is a 3D matrix (array) with dimenion sizes (x, y, spectral): 200 x 200 x 250
So this corresponds to 250 images, each having 200 x 200 pixels
The goal is to perform dimensionality reduction/cluster analysis and extract these five images and/or spectra
The various methods can be applied either to the image domain by treating the volume as a stack of images (250 images, each having 200 x 200 pixels), 
or in the spectral domain (200 x 200 spectra with 250 points in each spectrum)
'''


chemct = phantom5c_xrdct_images(npix, imAl, imCu, imFe, imPt, imZn)
print(chemct.shape)


#%% Visualise the hyperspectral volume

hs = hyperexpl.HyperSliceExplorer(chemct, np.arange(0,chemct.shape[2]), 'Channels')
hs.explore()



#%% We can try first using the scikit-learn library

from sklearn.decomposition import PCA, NMF, FastICA, LatentDirichletAllocation
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering

#%% PCA: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

pca = PCA(n_components=5).fit(data)
print(pca.components_.shape)

ii = 1
dp = pca.components_[ii,:]

plt.figure(1);plt.clf()
plt.plot(dp)
plt.show()

#%% PCA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

pca = PCA(n_components=5).fit(data)
print(pca.components_.shape)


ii = 1
im = pca.components_[ii,:]
im = np.reshape(im, (chemct.shape[0],chemct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()


#%% K means: images


data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

ims = np.zeros((chemct.shape[0], chemct.shape[1], clusters))
inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    ims[:,:,ii] = np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2)


ii = 3
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()

#%% AgglomerativeClustering: images


data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 5

ac = AgglomerativeClustering(distance_threshold=75, linkage="complete", n_clusters=None).fit(data)

labels = ac.labels_[:]

ims = np.zeros((chemct.shape[0], chemct.shape[1], np.max(labels)))
inds = []
for ii in range(np.max(labels)):

    inds.append(np.where(ac.labels_==ii))
    
    tmp = chemct[:,:,np.squeeze(inds[ii])]
    
    if len(tmp.shape)>2:
    
        ims[:,:,ii] = np.mean(tmp, axis = 2)
        
    else:
        
        ims[:,:,ii] = tmp

    print(ii)
    
#%%
ii = 5
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()

#%% Spectral clustering: images

spc = SpectralClustering(affinity="nearest_neighbors", n_clusters=5, eigen_solver="arpack").fit(data)

labels = spc.labels_[:]

ims = np.zeros((chemct.shape[0], chemct.shape[1], clusters))
inds = []
for ii in range(clusters):

    inds.append(np.where(spc.labels_==ii))
    
    ims[:,:,ii] = np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2)

#%
ii = 2
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()



#%% NMF: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

start = time.time()
nmf = NMF(n_components=10).fit(data+0.01)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

#%%
ii = 4
im = nmf.components_[ii,:]
im = np.reshape(im, (chemct.shape[0],chemct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()


#%% FastICA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

fica = FastICA(n_components=6).fit(data+0.0001)
print(nmf.components_.shape)

#%%
ii = 2
im = fica.components_[ii,:]
im = np.reshape(im, (chemct.shape[0],chemct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()



#%% LatentDirichletAllocation: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

lda = LatentDirichletAllocation(n_components=6, max_iter=5, learning_method="online", learning_offset=50.0, random_state=0).fit(data + 0.01)
print(lda.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot(lda.components_[ii,:]/np.max(lda.components_[ii,:]) + 0.05*ii)

plt.show()


#%% FastICA: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

fica = FastICA(n_components=6, random_state=0).fit(data)
print(fica.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot(fica.components_[ii,:]/np.max(fica.components_[ii,:]) + 0.05*ii)

plt.show()

#%% NMF: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

nmf = NMF(n_components=6).fit(data+0.01)
print(nmf.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot( nmf.components_[ii,:]/np.max(nmf.components_[ii,:]) + 0.05*ii)

plt.show()







#%% Let's add some noise!!!

im = np.mean(chemct, axis = 2)

# ct is count time (seconds)
imn = addpnoise2D(im, ct=500)

plt.figure(1);plt.clf()
plt.imshow(np.concatenate((im, imn), axis = 1), cmap = 'jet')
plt.colorbar()
plt.show()

#%% Noisy volume

start = time.time()
chemct_noisy = addpnoise3D(chemct, ct=500)
print('Adding noise took %s seconds' %(time.time() - start))

print(chemct_noisy.shape)


#%% Visualise the hyperspectral volume

hs = hyperexpl.HyperSliceExplorer(chemct_noisy, np.arange(0,chemct_noisy.shape[2]), 'Channels')
hs.explore()

#%% NMF: Images

data = np.reshape(chemct_noisy, (chemct_noisy.shape[0]*chemct_noisy.shape[1],chemct_noisy.shape[2])).transpose()

print(data.shape)

start = time.time()
nmf = NMF(n_components=6).fit(data+0.01)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

#%%
ii = 2
im = nmf.components_[ii,:]
im = np.reshape(im, (chemct_noisy.shape[0],chemct_noisy.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()





















