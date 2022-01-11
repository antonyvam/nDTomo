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

#% Ground truth

'''
These are the five ground truth componet spectra
'''

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()


'''
These are the five ground truth componet images
'''

npix = 150
imAl, imCu, imFe, imPt, imZn = phantom5c(npix)


#% Create the chemical tomography dataset

chemct = phantom5c_xrdct_images(npix, imAl, imCu, imFe, imPt, imZn)
print(chemct.shape)


#% Ground truth data

gtimlist = [imAl, imCu, imFe, imPt, imZn]
gtsplist = [dpAl, dpCu, dpFe, dpPt, dpZn]
gtldlist = ['Al', 'Cu', 'Fe', 'Pt', 'Zn']

#% Parameter search

methods = ['pca', 'hog', 'pca-hog']
embedding = 'tsne'
dim = (npix,npix)
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