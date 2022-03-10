# -*- coding: utf-8 -*-
"""
Dimensionality reduction/ cluster analysis using a phantom xrd-ct dataset

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import nDphantom_2D, load_example_patterns, nDphantom_3D, nDphantom_4D, nDphantom_2Dmap
from nDTomo.utils.misc import h5read_data, h5write_data, closefigs, showplot, showspectra, showim, showvol, normvol, addpnoise1D,  addpnoise2D, addpnoise3D, interpvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.ct.astra_tomo import astra_create_geo, astra_rec_vol, astra_rec_alg, astra_create_sino_geo, astra_create_sino
from nDTomo.ct.conv_tomo import radonvol, fbpvol

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, h5py

### Packages for clustering and dimensionality reduction

from sklearn.decomposition import PCA, NMF, FastICA, LatentDirichletAllocation
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from clustimage import Clustimage
import pymcr

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
spa = np.array(spectra)

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

#%% Now we will create a chemical-CT sinogram dataset using the astra tool box using GPU

nproj = 220

chemsinos = np.zeros((chemct.shape[0], nproj, chemct.shape[2]))

for ii in tqdm(range(chemct.shape[2])):
    
    proj_id = astra_create_sino_geo(chemct[:,:,ii], theta=np.deg2rad(np.arange(0, 180, 180/nproj)))
    chemsinos[:,:,ii] = astra_create_sino(chemct[:,:,ii], proj_id).transpose()

#%% We can also try with skimage and CPU which is very slow

nproj = 220 # Number of projections for the sinogram data

chemsinos = radonvol(chemct, scan=180, theta = np.arange(0, 180, 180/nproj))

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(chemsinos)
hs.explore()

#%% Finally let's create a 2D chemical map after taking a projection from a 3D chemical-CT dataset

vol4d = nDphantom_4D(npix = 200, nzt = 100, vtype = 'Spectral', indices = 'Random', spectra=spectra, imgs=iml, norm = 'Volume')

print('The volume dimensions are %d, %d, %d, %d' %(vol4d.shape[0], vol4d.shape[1], vol4d.shape[2], vol4d.shape[3]))

#%% Now create a projection dataset from the 3D chemical-ct dataset

map2D = nDphantom_2Dmap(vol4d, dim = 0)

print('The map dimensions are %d, %d, %d' %(map2D.shape[0], map2D.shape[1], map2D.shape[2]))

#%%

hs = HyperSliceExplorer(map2D.transpose(1,0,2))
hs.explore()

#%% Export the simulated data

p = 'data\\'
fn = 'phantom_data'

h5write_data(p, fn, ['ground_truth_images', 'ground_truth_spectra', 'tomo', 'sinograms', 'map'], [gtimlist, gtsplist, chemct, chemsinos, map2D])





#%%

'''
Part 2: Data analysis
'''

#%% Let's load the data

# Specify the path to the data
p = 'data\\'
fn = 'phantom_data'

data = h5read_data(p, fn,  ['ground_truth_images', 'ground_truth_spectra', 'tomo', 'sinograms', 'map'])
print(data.keys())

#%%

gtimlist = data['ground_truth_images']
gtsplist = data['ground_truth_spectra']
chemct = data['tomo']
chemsinos = data['sinograms']
map2D = data['map']

#%% PCA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

imagelist, legendlist = create_complist_imgs(pca.components_, chemct.shape[0], chemct.shape[1])

imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []; gtldlist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii].transpose())
    llist.append(legendlist[ii])
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
    clist.append(gtimlist[ii].transpose())
    llist.append(legendlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))



plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)


#%% Do an MCR analysis using noiseless data and bad initial guesses

mcrar = pymcr.mcr.McrAR(max_iter=50, st_regr='NNLS', 
                st_constraints=[pymcr.constraints.ConstraintNonneg()])

initial_spectra = addpnoise1D(np.copy(spa)+1E-10, 250)

mcrar.fit(chemct.reshape((chemct.shape[0]*chemct.shape[1], chemct.shape[2]))+1E-10, ST=initial_spectra, verbose=True)
print('\nFinal MSE: {:.7e}'.format(mcrar.err[-1]))


#%%

plt.figure(1);plt.clf()
plt.plot(spa.transpose());
plt.title('Ground truth')
plt.show()

plt.figure(2);plt.clf()
plt.plot(initial_spectra.transpose());
plt.title('Initial guess')
plt.show()

plt.figure(3);plt.clf()
plt.plot(mcrar.ST_opt_.T);
plt.title('MCA retrieved')
plt.show()

plt.figure(4);plt.clf()
plt.plot(mcrar.ST_opt_.T - spa.transpose());
plt.title('Difference')
plt.show()

#%% Do an MCR analysis using noisy data and bad initial guesses

mcrar = pymcr.mcr.McrAR(max_iter=50, st_regr='NNLS', 
                st_constraints=[pymcr.constraints.ConstraintNonneg()])


sp = np.sum(np.sum(chemct, axis =0), axis=0)
sp = sp/np.max(sp)
initial_spectra = np.tile(sp, (5,1))
# initial_spectra = addpnoise1D(np.copy(spa)+1E-10, 1E4)

initial_spectra[0:3,:] = addpnoise1D(np.copy(spa[0:3,:])+1E-10, 1E4)

# Add different noise per spectrum
for ii in range(initial_spectra.shape[0]):
    ct = np.random.randint(0,1990)
    print(ct)
    initial_spectra[ii,:] = addpnoise1D(initial_spectra[ii,:], 10 + ct)

chemctn = np.copy(chemct)+1E-10
chemctn = addpnoise3D(chemctn, 50)

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(chemctn)
hs.explore()

#%%

chemctn = chemctn.reshape((chemctn.shape[0]*chemctn.shape[1], chemctn.shape[2]))

mcrar.fit(chemctn, ST=initial_spectra, verbose=True)
print('\nFinal MSE: {:.7e}'.format(mcrar.err[-1]))


#%%

plt.figure(1);plt.clf()
plt.plot(spa.transpose());
plt.title('Ground truth')
plt.show()

plt.figure(2);plt.clf()
plt.plot(initial_spectra.transpose());
plt.title('Initial guess')
plt.show()

plt.figure(3);plt.clf()
plt.plot(mcrar.ST_opt_.T);
plt.title('MCA retrieved')
plt.show()

plt.figure(4);plt.clf()
plt.plot(mcrar.ST_opt_.T - spa.transpose());
plt.title('Difference')
plt.show()

#%%

mcrims = mcrar.C_opt_.reshape((chemct.shape[0], chemct.shape[1], initial_spectra.shape[0]))

ii = 4

plt.figure(5);plt.clf()
plt.imshow(mcrims[:,:,ii]);
plt.colorbar()
plt.show()


#%% AgglomerativeClustering: images


data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

clusters = 5

ac = AgglomerativeClustering(distance_threshold=75, linkage="complete", n_clusters=None).fit(data)

labels = ac.labels_[:]

imagelist = []; inds = []
for ii in range(np.max(labels)):

    inds.append(np.where(ac.labels_==ii))
    
    tmp = chemct[:,:,np.squeeze(inds[ii])]
    
    if len(tmp.shape)>2:
    
        imagelist.append(np.mean(tmp, axis = 2))
        
    else:
        
        imagelist.append(tmp)

    print(ii)
    
imagelist = [imagelist[4], imagelist[1], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Cluster %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)



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

imagelist, legendlist = create_complist_imgs(nmf.components_, chemct.shape[0], chemct.shape[1])

imagelist = [imagelist[1], imagelist[4], imagelist[2], imagelist[3], imagelist[0]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Component %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)

#%%


start = time.time()
nmf = NMF(n_components=10).fit_transform(data.transpose()+0.01, W=None, H=None)
print('NMF analysis took %s seconds' %(time.time() - start))
print(nmf.shape)

#%%
ii = 0
im = nmf[:,ii]
im = im.reshape(chemct.shape[0],chemct.shape[1])

showim(im, 1)

#%%

start = time.time()
nmf2 = NMF(n_components=10, init='custom').fit_transform(data.transpose()+0.01, W=np.copy(nmf), H=np.ones((10, 250)))
print('NMF analysis took %s seconds' %(time.time() - start))
print(nmf.shape)

#%%
ii = 3
im = nmf2[:,ii].reshape(chemct.shape[0],chemct.shape[1])

showim(im, 1)

#%% FastICA: Images

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2])).transpose()

print(data.shape)

fica = FastICA(n_components=5).fit(data+0.0001)
print(fica.components_.shape)

imagelist, legendlist = create_complist_imgs(fica.components_, chemct.shape[0], chemct.shape[1])

# imagelist = [imagelist[0], imagelist[4], imagelist[1], imagelist[2], imagelist[3]]

clist = []; llist = []
for ii in range(len(gtimlist)):
    clist.append(gtimlist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(imagelist)):
    clist.append(imagelist[ii])
    llist.append('Component %d' %(ii + 1))


plotfigs_imgs(clist, llist, rows=2, cols=5, figsize=(20,6), cl=True)





#%% PCA: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

start = time.time()
pca = PCA(n_components=5).fit(data)
print('PCA analysis took %s seconds' %(time.time() - start))

print(pca.components_.shape)

spectralist, legendlist = create_complist_spectra(pca.components_)

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Component %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


#%% K means: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

kmeans = KMeans(init="k-means++", n_clusters=clusters, n_init=4, random_state=0).fit(data)

labels = kmeans.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(kmeans.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


#%% AgglomerativeClustering: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

ac = AgglomerativeClustering(n_clusters=clusters).fit(data)

labels = ac.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(ac.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))

#%% SpectralClustering: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

clusters = 5

spc = SpectralClustering(affinity="nearest_neighbors", n_clusters=5, eigen_solver="arpack").fit(data)

labels = spc.labels_[:]

spectralist = []; inds = []
for ii in range(clusters):

    inds.append(np.where(ac.labels_==ii))
    
    spectralist.append(np.mean(data[np.squeeze(inds[ii]),:], axis = 0))

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Cluster %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))

#%% FastICA: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

fica = FastICA(n_components=6, random_state=0).fit(data)
print(fica.components_.shape)

spectralist, legendlist = create_complist_spectra(fica.components_)

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Component %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))


#%% NMF: Spectra

data = np.reshape(chemct, (chemct.shape[0]*chemct.shape[1],chemct.shape[2]))

print(data.shape)

nmf = NMF(n_components=6).fit(data+0.01)
print(nmf.components_.shape)

spectralist, legendlist = create_complist_spectra(nmf.components_)

spectralist = [spectralist[1], spectralist[4], spectralist[2], spectralist[3], spectralist[0]]

slist = []; llist = []
for ii in range(len(gtimlist)):
    slist.append(gtsplist[ii])
    llist.append(gtldlist[ii])
    
for ii in range(len(spectralist)):
    slist.append(spectralist[ii])
    llist.append('Component %d' %(ii + 1))
    
plotfigs_spectra(slist, llist, xaxis=np.arange(0,slist[0].shape[0]), rows=2, cols=5, figsize=(20,7))



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

hs = HyperSliceExplorer(chemct_noisy)
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





































