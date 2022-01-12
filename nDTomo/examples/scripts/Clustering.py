# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import phantom5c_xanesct, phantom5c_xrdct, load_example_patterns, phantom5c, phantom5c_xrdct_images
from nDTomo.utils import hyperexpl
from nDTomo.utils.misc import addpnoise2D, addpnoise3D, interpvol, showplot, showim, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
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

#%% Ground truth data

gtimlist = [imAl, imCu, imFe, imPt, imZn]
gtsplist = [dpAl, dpCu, dpFe, dpPt, dpZn]
gtldlist = ['Al', 'Cu', 'Fe', 'Pt', 'Zn']

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

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

spectralist, legendlist = create_complist_spectra(pca.components_)

plotfigs_spectra(spectralist, legendlist, xaxis=np.arange(0,spectralist[0].shape[0]), rows=1, cols=5, figsize=(20,3))


#%% PCA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

imagelist, legendlist = create_complist_imgs(pca.components_, chemct.shape[0], chemct.shape[1])

imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append(legendlist[ii])


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)


#%% K means: images


data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

imagelist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    imagelist.append(np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2))


imagelist = [imagelist[2], imagelist[4], imagelist[3], imagelist[0], imagelist[1]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))



plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

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

imagelist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(spc.labels_==ii))
    
    imagelist.append(np.mean(chemct[:,:,np.squeeze(inds[ii])], axis = 2))

# imagelist = [imagelist[3], imagelist[0], imagelist[1], imagelist[4], imagelist[2]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)


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

#%% Clustimage

'''
Class initialisation:
    
    Parameters
    ----------
    method : str, (default: 'pca')
        Method to be usd to extract features from images.
            * None : No feature extraction
            * 'pca' : PCA feature extraction
            * 'hog' : hog features extraced
            * 'pca-hog' : PCA extracted features from the HOG desriptor
            hashmethod : str (default: 'ahash')
            * 'ahash': Average hash
            * 'phash': Perceptual hash
            * 'dhash': Difference hash
            * 'whash-haar': Haar wavelet hash
            * 'whash-db4': Daubechies wavelet hash
            * 'colorhash': HSV color hash
            * 'crop-resistant': Crop-resistant hash
    embedding : str, (default: 'tsne')
        Perform embedding on the extracted features. The xycoordinates are used for plotting purposes.
            * 'tsne' or  None
    grayscale : Bool, (default: False)
        Colorscaling the image to gray. This can be usefull when clustering e.g., faces.
    dim : tuple, (default: (128,128))
        Rescale images. This is required because the feature-space need to be the same across samples.
    dirpath : str, (default: None)
        Directory to write images.
    ext : list, (default: ['png','tiff','jpg'])
        Images with the file extentions are used.
    params_pca : dict, default: {'n_components':50, 'detect_outliers':None}
        Parameters to initialize the pca model.
    params_hog : dict, default: {'orientations':9, 'pixels_per_cell':(16,16), 'cells_per_block':(1,1)}
        Parameters to extract hog features.
    verbose : int, (default: 20)
        Print progress to screen. The default is 20.
        60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

fit_transform method:
    
    Parameters
    ----------
    X : [str of list] or [np.array].
        The input can be:
            * "c://temp//" : Path to directory with images
            * ['c://temp//image1.png', 'c://image2.png', ...] : List of exact pathnames.
            * [[.., ..], [.., ..], ...] : np.array matrix in the form of [sampels x features]
    cluster : str, (default: 'agglomerative')
        Type of clustering.
            * 'agglomerative'
            * 'kmeans'
            * 'dbscan'
            * 'hdbscan'
    evaluate : str, (default: 'silhouette')
        Cluster evaluation method.
            * 'silhouette'
            * 'dbindex'
            * 'derivative'
    metric : str, (default: 'euclidean').
        Distance measures. All metrics from sklearn can be used such as:
            * 'euclidean'
            * 'hamming'
            * 'cityblock'
            * 'correlation'
            * 'cosine'
            * 'jaccard'
            * 'mahalanobis'
            * 'seuclidean'
            * 'sqeuclidean'
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
            * 'ward'
            * 'single'
            * 'complete'
            * 'average'
            * 'weighted'
            * 'centroid'
            * 'median'
    min_clust : int, (default: 3)
        Number of clusters that is evaluated greater or equals to min_clust.
    max_clust : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to max_clust.
    cluster_space: str, (default: 'high')
        Selection of the features that are used for clustering. This can either be on high or low feature space.
            * 'high' : Original feature space.
            * 'low' : Input are the xycoordinates that are determined by "embedding". Thus either tSNE coordinates or the first two PCs or HOGH features.
    black_list : list, (default: None)
        Exclude directory with all subdirectories from processing.
        * example: ['undouble']
'''

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

# init
cl = Clustimage()

results = cl.fit_transform(data)

print(results.keys())

cl.cluster(min_clust=4, max_clust=15)

# Get the unique images
unique_samples = cl.unique()
# 
print(unique_samples.keys())

data[unique_samples['idx'],:]

cl.plot_unique()

#%%
imlist = []; legendlist = []
for ii in range(len(unique_samples['labels'])):
    
    imlist.append(np.mean(unique_samples['img_mean'][ii].reshape(chemct.shape[0],chemct.shape[1],3), axis = 2)/255)
    legendlist.append('Component %d' %(ii + 1))

plotfigs_imgs(imlist, legendlist, rows=2, cols=5, figsize=(20,9), cl=True)



#%% Parameter search

methods = ['pca', 'hog', 'pca-hog']
embedding = 'tsne'
dim = (200,200)
params_pca = {'n_components':50, 'detect_outliers':None}
params_hog = {'orientations':9, 'pixels_per_cell':(16,16), 'cells_per_block':(1,1)}

clusters = ['agglomerative', 'kmeans', 'dbscan', 'hdbscan']
evaluations = ['silhouette', 'dbindex','derivative']
metrics = ['euclidean','hamming','cityblock','correlation','cosine','jaccard',
          'mahalanobis','seuclidean','sqeuclidean']
link = ['ward','single','complete','average','weighted','centroid','median']  
min_clust = 3
max_clust = 25
cluster_spaces = ['high', 'low']
        
kk = 0

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

for method in methods:
    
    for cluster in clusters:
        
        for evaluation in evaluations:
            
            for metric in metrics:
                
                for linkage in link:
                    
                    for cluster_space in cluster_spaces:

                        # init
                        cl = Clustimage(method=method, embedding='tsne', dim=dim, 
                                        params_pca=params_pca, params_hog=params_hog)
                        
                        results = cl.fit_transform(X=data, cluster=cluster, evaluate=evaluation,
                                                   metric=metric, linkage=linkage,
                                                   min_clust=min_clust, max_clust=max_clust,
                                                   cluster_space = cluster_space)
                        
                        # Get the unique images
                        unique_samples = cl.unique()
                        #                         
                        data[unique_samples['idx'],:]
                        
                        imlist = []; legendlist = []
                        for ii in range(len(unique_samples['labels'])):
                            
                            imlist.append(np.mean(unique_samples['img_mean'][ii].reshape(chemct.shape[0],chemct.shape[1],3), axis = 2)/255)
                            legendlist.append('Component %d' %(ii + 1))

                        # Save the results
                        
                        fn = 'phantom_noiseless_%d.h5' %kk
                        print(fn)
                        
                        with h5py.File(fn, 'w') as f:
                            
                            f.create_dataset('results', data = data[unique_samples['idx'],:].reshape(len(unique_samples['labels']), chemct.shape[0], chemct.shape[1]))
                            f.create_dataset('components', data = imlist)
                            f.create_dataset('cluster', data = cluster)
                            f.create_dataset('method', data = method)
                            f.create_dataset('evaluate', data = evaluation)
                            f.create_dataset('metric', data = metric)
                            f.create_dataset('linkage', data = linkage)
                            f.create_dataset('cluster_space', data = cluster_space)

                        f.close()

                        kk = kk + 1



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





#%% Experimental data - POX catalyst data

'''
Download the POX experimental data from zenodo: https://zenodo.org/record/4664597
'''

import urllib
urllib.urlretrieve('https://zenodo.org/record/4664597/files/Multiphase_experimental_test_data.h5', 'Multiphase_experimental_test_data.h5')

#%% Load the data and prepare them from statistical analysis

fn = 'C:\\Users\\Admin\\Downloads\\Multiphase_experimental_test_data.h5'

with h5py.File(fn, 'r') as f:
    print(f.keys())
    vol = np.array(f['Patterns'][:])

vol = np.transpose(vol, (2,1,0))

print(vol.shape)
vol = np.where(vol<0, 0, vol)

showplot(np.sum(np.sum(vol,axis=0), axis = 0), 1)

#%% Volume pre-processing

#### We can crop a bit the chemical volume
# roi = np.arange(100, 350)
# vol = vol[:,:,roi]

#### Normalise the volume with respect to the max value
vol = 100*vol/np.max(vol)

#### We can normalise all images
# vol = normvol(vol)

#%% NMF: Images/Sinograms

data = np.reshape(vol, (vol.shape[0]*vol.shape[1],vol.shape[2])).transpose()

print(data.shape)

start = time.time()
nmf = NMF(n_components=15).fit(data+0.001)
print('NMF analysis took %s seconds' %(time.time() - start))

print(nmf.components_.shape)

imagelist, legendlist = create_complist_imgs(nmf.components_, vol.shape[0], vol.shape[1])

plotfigs_imgs(imagelist, legendlist, rows=3, cols=5, figsize=(20,9), cl=True)

#%% NMF: Spectra

data = np.reshape(vol, (vol.shape[0]*vol.shape[1],vol.shape[2]))

print(data.shape)

nmf = NMF(n_components=15).fit(data+0.001)
print(nmf.components_.shape)

spectralist, legendlist = create_complist_spectra(nmf.components_)

plotfigs_spectra(spectralist, legendlist, xaxis=np.arange(0,spectralist[0].shape[0]), rows=3, cols=5, figsize=(20,6))

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





































