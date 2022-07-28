# -*- coding: utf-8 -*-
"""
Tests for the various nDTomo phantoms

@author: Kenan Gnonhoue and Antony Vamvakeros
"""

#%% First let's import some necessary modules

from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.sim.shapes.phantoms import SheppLogan
from nDTomo.utils.misc import closefigs, addpnoise2D, addpnoise3D, interpvol, showplot, showspectra, showim, normvol, plotfigs_imgs, plotfigs_spectra, create_complist_imgs, create_complist_spectra
from nDTomo.utils.misc3D import showvol
from nDTomo.ct.astra_tomo import astra_create_sino, astra_rec_single

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, h5py

%matplotlib auto

#%% Create a Shepp-Logan phantom

npix = 256
im = SheppLogan(npix)

showim(im, 1, cmap='jet')

#%% Create sinogram

sino = astra_create_sino(im)

showim(sino, 2, cmap='jet')

rec = astra_rec_single(sino.transpose())

showim(rec, 3, cmap='jet')

#%% Noisy sinogram

sino_noisy = addpnoise2D(sino, ct=10)

showim(sino_noisy, 2, cmap='jet')

rec = astra_rec_single(sino_noisy.transpose())

showim(rec, 3, cmap='jet')

#%% Sinogram with angular undersampling

sino_under = astra_create_sino(im, npr=256/4)

showim(sino_under, 2, cmap='jet')

rec = astra_rec_single(sino_under.transpose())

showim(rec, 3, cmap='jet')

#%% Noisy sinogram with angular undersampling

sino_under = astra_create_sino(im, npr=256/4)
sino_under_noisy = addpnoise2D(sino_under, ct=10)

showim(sino_under_noisy, 2, cmap='jet')

rec = astra_rec_single(sino_under_noisy.transpose())

showim(rec, 3, cmap='jet')


#%% Let's try some other methods than FBP for the sinogram that has angular undersampling!


rec = astra_rec_single(sino_under.transpose(), method='SIRT_CUDA', nits=200)
# rec = astra_rec_single(sino_under.transpose(), method='SART_CUDA', nits=200)
# rec = astra_rec_single(sino_under.transpose(), method='ART', nits=100000)
# rec = astra_rec_single(sino_under.transpose(), method='CGLS', nits=20)

showim(rec, 1, cmap='jet')


#%% Let's try some other methods than FBP for the noisy sinogram!


rec = astra_rec_single(sino_noisy.transpose(), method='SIRT_CUDA', nits=200)
# rec = astra_rec_single(sino_noisy.transpose(), method='SART_CUDA', nits=200)
# rec = astra_rec_single(sino_noisy.transpose(), method='ART', nits=100000)
# rec = astra_rec_single(sino_noisy.transpose(), method='CGLS', nits=20)

showim(rec, 1, cmap='jet')
