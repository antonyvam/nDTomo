# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:53:21 2022

@author: Antony Vamvakeros
"""

#%%

from nDTomo.sim.shapes.phantoms import phantom_3Dxrdct, phantom_microct, phantom5c_microct, phantom5c_3Dxrdct, phantom5c_xanesct, phantom5c_xrdct, load_example_patterns, phantom5c, phantom5c_xrdct
from nDTomo.utils import hyperexpl
from nDTomo.utils.misc import addpnoise2D, addpnoise3D, interpvol, showplot, showim, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
import numpy as np
import matplotlib.pyplot as plt
import time, h5py
from mayavi import mlab

#%% Ground truth

'''
These are the five ground truth componet spectra
'''

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()

'''
These are the five ground truth componet images
'''

npix = 150
imAl, imCu, imFe, imPt, imZn = phantom5c(npix)

#%%

chemct = phantom5c_xrdct(npix, imgs = [imAl, imCu, imFe, imPt, imZn], dps = [dpAl, dpCu, dpFe, dpPt, dpZn])
print(chemct.shape)

#%%

xrdct3d = phantom5c_3Dxrdct(npix, nz = 5, imgs = [imAl, imCu, imFe, imPt, imZn], dps = [dpAl, dpCu, dpFe, dpPt, dpZn])
print(xrdct3d.shape)

#%%
# microct = phantom_microct(npix=100, nz = 100, imgs = None)
m2 = phantom5c_microct(npix=100, imgs = None, dps = None)

#%%

hs = hyperexpl.HyperSliceExplorer(xrdct3d[:,:,:,62], np.arange(0,xrdct3d.shape[2]), 'Channels')
hs.explore()

#%%
mlab.pipeline.volume(mlab.pipeline.scalar_field(m2)) #, vmin=0, vmax=0.8



#%%

xrdct3d = phantom_3Dxrdct(npix = 50, nz = 50)
print(xrdct3d.shape)

#%%

mlab.pipeline.volume(mlab.pipeline.scalar_field(xrdct3d[:,:,:,89])) #, vmin=0, vmax=0.8

#%%

hs = hyperexpl.HyperSliceExplorer(xrdct3d[:,:,6,:], np.arange(0,xrdct3d.shape[3]), 'Channels')
hs.explore()










