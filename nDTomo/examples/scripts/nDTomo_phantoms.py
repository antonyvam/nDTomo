# -*- coding: utf-8 -*-
"""
Tests for the various nDTomo phantoms

@author: Antony Vamvakeros
"""

#%% First let's import some necessary modules

from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.sim.shapes.phantoms import nDphantom_2D, nDphantom_3D, nDphantom_4D, nDphantom_5D, nDphantom_2Dmap
from nDTomo.utils.misc import closefigs, addpnoise2D, addpnoise3D, interpvol, showplot, showspectra, showim, showvol, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
import numpy as np


#%% Let's create a 2D image and get the components used to create it

# This creates a single image, npix specifies the number of pixes in the output image
# The image is normalised

imc = nDphantom_2D(npix = 100, nim = 'One')
print('The image dimensions are %d, %d' %(imc.shape[0], imc.shape[1]))

# Let's plot the image
showim(imc)

# This creates a list containing five images, all with the same dimensions
iml = nDphantom_2D(npix = 200, nim = 'Multiple')
print(len(iml))


plotfigs_imgs(iml)

#%% Close the various figures

closefigs()

#%% Let's create a 3D (micro-CT) dataset with three spatial dimensions (x,y,z)

# This will create a volume dataset with dimension sizes: (256, 256, 300)

vol = nDphantom_3D(npix=256, nz = 300, indices = 'Random', norm = 'Volume')

print('The volume dimensions are %d, %d, %d' %(vol.shape[0], vol.shape[1], vol.shape[2]))

#%% Let's visualise the volume using an orthoslice

hs = HyperSliceExplorer(vol)
hs.explore()

#%% Let's visualise the volume using an orthoslice at a different plane

hs = HyperSliceExplorer(vol.transpose(2,0,1))
hs.explore()

#%% Let's visualise the volume using an orthoslice at a different plane

hs = HyperSliceExplorer(vol.transpose(2,1,0))
hs.explore()

#%% Let's perform a volume rendering

showvol(vol, [0, 0.75])


#%% Let's create a 3D (chemical-CT) dataset with two spatial and one spectral dimensions (x,y,spectral)

# This will create a volume dataset with dimension sizes: (256, 256, 250)

vol = nDphantom_3D(npix=256, use_spectra = 'Yes', indices = 'All',  norm = 'No')

print('The volume dimensions are %d, %d, %d' %(vol.shape[0], vol.shape[1], vol.shape[2]))

#%% Let's explore the local patterns and chemical images

hs = HyperSliceExplorer(vol)
hs.explore()

#%% Let's perform a volume rendering

showvol(vol)


#%% Let's create a 4D dataset with three spatial and one temporal dimensions (x,y,z,time)

vol = nDphantom_4D(npix = 200, nzt = 5, vtype = 'Temporal', indices = 'Random',  norm = 'Volume')

print('The volume dimensions are %d, %d, %d, %d' %(vol.shape[0], vol.shape[1], vol.shape[2], vol.shape[3]))


#%% Let's explore the data using an orthoslice for the five volumes

hs = HyperSliceExplorer(np.concatenate((vol[:,:,:,0], vol[:,:,:,1], vol[:,:,:,2], vol[:,:,:,3], vol[:,:,:,4]), axis = 1))
hs.explore()

#%% Let's perform a volume rendering

showvol(np.concatenate((vol[:,:,:,0], vol[:,:,:,1], vol[:,:,:,2], vol[:,:,:,3], vol[:,:,:,4]), axis = 1))


#%% Let's create a 4D dataset with three spatial and one spectral dimensions (x,y,z,chemistry)

vol = nDphantom_4D(npix = 200, nzt = 100, vtype = 'Spectral', indices = 'Random',  norm = 'Volume')

print('The volume dimensions are %d, %d, %d, %d' %(vol.shape[0], vol.shape[1], vol.shape[2], vol.shape[3]))

#%%

hs = HyperSliceExplorer(np.concatenate((vol[:,:,0,:], vol[:,:,1,:], vol[:,:,2,:], vol[:,:,3,:], vol[:,:,4,:]), axis = 1))
hs.explore()

#%% Now create a projection dataset from the 3D chemical-ct dataset

map2D = nDphantom_2Dmap(vol, dim = 0)

print('The map dimensions are %d, %d, %d' %(map2D.shape[0], map2D.shape[1], map2D.shape[2]))

#%%

hs = HyperSliceExplorer(map2D.transpose(1,0,2))
hs.explore()






















































