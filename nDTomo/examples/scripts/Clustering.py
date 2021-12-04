# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:14:52 2021

@author: Antony
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF, FastICA, LatentDirichletAllocation
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
import h5py


#%% Load the dataset

fn = 'Phantom_xrdct.h5'

with h5py.File(fn, 'r') as f:
    xrdct = np.array(f['data'][:])

#%% PCA: Spectra

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2]))

print(data.shape)

pca = PCA(n_components=5).fit(data)
print(pca.components_.shape)

ii = 1
dp = pca.components_[ii,:]

plt.figure(1);plt.clf()
plt.plot(dp)
plt.show()

#%% PCA: Images

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2])).transpose()

print(data.shape)

pca = PCA(n_components=5).fit(data)
print(pca.components_.shape)


ii = 1
im = pca.components_[ii,:]
im = np.reshape(im, (xrdct.shape[0],xrdct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()


#%% K means: images


data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2])).transpose()

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

ims = np.zeros((xrdct.shape[0], xrdct.shape[1], clusters))
inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    ims[:,:,ii] = np.mean(xrdct[:,:,np.squeeze(inds[ii])], axis = 2)


ii = 4
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()

#%% AgglomerativeClustering: images


data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2])).transpose()

print(data.shape)

clusters = 5

ac = AgglomerativeClustering(distance_threshold=75, linkage="complete", n_clusters=None).fit(data)

labels = ac.labels_[:]

ims = np.zeros((xrdct.shape[0], xrdct.shape[1], np.max(labels)))
inds = []
for ii in range(np.max(labels)):

    inds.append(np.where(ac.labels_==ii))
    
    tmp = xrdct[:,:,np.squeeze(inds[ii])]
    
    if len(tmp.shape)>2:
    
        ims[:,:,ii] = np.mean(tmp, axis = 2)
        
    else:
        
        ims[:,:,ii] = tmp

    print(ii)
    
#%%
ii = 1
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()

#%% Spectral clustering: images

spc = SpectralClustering(affinity="nearest_neighbors", n_clusters=5, eigen_solver="arpack").fit(data)

labels = spc.labels_[:]

ims = np.zeros((xrdct.shape[0], xrdct.shape[1], clusters))
inds = []
for ii in range(clusters):

    inds.append(np.where(spc.labels_==ii))
    
    ims[:,:,ii] = np.mean(xrdct[:,:,np.squeeze(inds[ii])], axis = 2)

#%
ii = 4
im = ims[:,:,ii]

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()





#%% NMF: Images

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2])).transpose()

print(data.shape)

nmf = NMF(n_components=6).fit(data+0.0001)
print(nmf.components_.shape)

#%%
ii = 0
im = nmf.components_[ii,:]
im = np.reshape(im, (xrdct.shape[0],xrdct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()


#%% FastICA: Images

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2])).transpose()

print(data.shape)

fica = FastICA(n_components=6).fit(data+0.0001)
print(nmf.components_.shape)

#%%
ii = 2
im = fica.components_[ii,:]
im = np.reshape(im, (xrdct.shape[0],xrdct.shape[1]))

plt.figure(1);plt.clf()
plt.imshow(im)
plt.colorbar()
plt.show()





#%% LatentDirichletAllocation: Spectra

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2]))

print(data.shape)

lda = LatentDirichletAllocation(n_components=6, max_iter=5, learning_method="online", learning_offset=50.0, random_state=0).fit(data + 0.0001)
print(lda.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot(q, lda.components_[ii,:]/np.max(lda.components_[ii,:]) + 0.05*ii)

plt.show()


#%% FastICA: Spectra

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2]))

print(data.shape)

fica = FastICA(n_components=6, random_state=0).fit(data)
print(fica.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot(q, fica.components_[ii,:]/np.max(fica.components_[ii,:]) + 0.05*ii)

plt.show()

#%% NMF: Spectra

data = np.reshape(xrdct, (xrdct.shape[0]*xrdct.shape[1],xrdct.shape[2]))

print(data.shape)

nmf = NMF(n_components=6).fit(data+0.0001)
print(nmf.components_.shape)

#%%

plt.figure(2);plt.clf()
for ii in range(5):
    
    plt.plot(q, nmf.components_[ii,:]/np.max(nmf.components_[ii,:]) + 0.05*ii)

plt.show()

#%%

plt.figure(1);plt.clf()

plt.plot(q, dpAl)
plt.plot(q, dpPt+0.05*1)
plt.plot(q, dpCu+0.05*2)
plt.plot(q, dpFe+0.05*3)
plt.plot(q, dpZn+0.05*4)





















