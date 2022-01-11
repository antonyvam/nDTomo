# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import phantom5c_xanesct, phantom5c_xrdct, load_example_patterns, phantom5c, phantom5c_xrdct_images
from nDTomo.utils import hyperexpl
from nDTomo.utils.misc import addpnoise2D, addpnoise3D, interpvol, showplot, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
import numpy as np
import matplotlib.pyplot as plt
import time, h5py

### Packages for clustering and dimensionality reduction

from sklearn.decomposition import PCA, NMF, FastICA, LatentDirichletAllocation
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from clustimage import Clustimage
from hyperspy.signals import Signal1D, Signal2D

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

#%%

plt.close('all')

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


#%% Interpolate in the spectral dimension - useful to test the timings of various methods as a function of spectral size

nchan = 1000
chemcti = interpvol(chemct, xold=np.linspace(0, chemct.shape[2], chemct.shape[2]), xnew=np.linspace(0, chemct.shape[2], nchan))
print(chemcti.shape)

#%% Visualise the hyperspectral volume

hs = hyperexpl.HyperSliceExplorer(chemct, np.arange(0,chemct.shape[2]), 'Channels')
hs.explore()

#%%

s = Signal1D(chemct+0.01)

# s.decomposition(algorithm="SVD")
s.decomposition(algorithm="NMF", output_dimension=10)
# s.decomposition(algorithm="MLPCA", output_dimension=10)

factors = s.get_decomposition_factors()

print(factors.data.shape, len(factors))

# s.decomposition(algorithm="NMF")

s.plot_decomposition_results()


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

# ica = FastICA()
# S_ = ica.fit_transform(data)
# A_ = ica.mixing_.T

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


#%% Clustimage


# init
cl = Clustimage()

results = cl.fit_transform(data)

print(results.keys())


# Get the unique images
unique_samples = cl.unique()
# 
print(unique_samples.keys())

data[unique_samples['idx'],:]


cl.plot_unique()


#%%

# No images in scatterplot
cl.scatter(zoom=None)



#%% Experimental data

datasets = ['Mn_catalyst_ch4149_rt', 'BCFZ_Mn_catalyst_ch4149_rt_abscor',
            'SOFC2_fresh_z-4.000', 'NMC532_pristine',
            'POXhr']

datasets = ['Mn_catalyst_ch4149_rt', 'BCFZ_Mn_catalyst_ch4149_rt',
            'SOFC2_fresh_z-4.000', 'NMC532_pristine',
            'POXhr']

p = 'Y:\\Development\\'

dd = 1

fn = '%s%s_rec.h5' %(p, datasets[dd])
fn = '%s%s_sinograms.h5' %(p, datasets[dd])

# fn = 'D:\\Dropbox (Finden)\\Finden_Research\\Legacy_Projects\\Beamtime_Names\\IHMA120\\reconstructions\\xrdct_2_denoised.h5'

with h5py.File(fn, 'r') as f:
    vol = np.array(f['data'][:])

vol = np.transpose(vol, (2,1,0))

print(vol.shape)

vol = np.where(vol<0, 0, vol)

showplot(np.sum(np.sum(vol,axis=0), axis = 0), 1)


#%% Volume pre-processing

# We can crop a bit the chemical volume
roi = np.arange(100, 350)
roi = np.arange(100, 510)
# roi = np.arange(300, 550)
vol = vol[:,:,roi]

# Normalise the volume with respect to the max value
vol = 100*vol/np.max(vol)

# We can normalise all images
# vol = normvol(vol)

#%% NMF: Images/Sinograms

data = np.reshape(vol, (vol.shape[0]*vol.shape[1],vol.shape[2])).transpose()

print(data.shape)

start = time.time()
nmf = NMF(n_components=10).fit(data+0.001)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

#%%


imagelist, legendlist = create_complist_imgs(nmf.components_, vol.shape[0], vol.shape[1])

plotfigs_imgs(imagelist, legendlist, rows=2, cols=5, figsize=(20,6), cl=True)


#%% NMF: Spectra

data = np.reshape(vol, (vol.shape[0]*vol.shape[1],vol.shape[2]))

print(data.shape)

nmf = NMF(n_components=10).fit(data+0.001)
print(nmf.components_.shape)

#%%

spectralist, legendlist = create_complist_spectra(nmf.components_)

plotfigs_spectra(spectralist, legendlist, xaxis=np.arange(0,spectralist[0].shape[0]), rows=2, cols=5, figsize=(20,6))

#%% Cluster image analysis

data = np.reshape(vol, (vol.shape[0]*vol.shape[1],vol.shape[2])).transpose()

# init
cl = Clustimage()

results = cl.fit_transform(data)

print(results.keys())

cl.cluster(min_clust=5, max_clust=15)

# Get the unique images
unique_samples = cl.unique()
# 
print(unique_samples.keys())

data[unique_samples['idx'],:]


cl.plot_unique()





































