# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:51:35 2023

@author: Antony Vamvakeros
"""



#%%

import torch
import torch.optim as optim
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from torch.nn.functional import grid_sample

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%

import h5py
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure, binary_erosion
import astra

from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.sim.shapes.phantoms import nDphantom_3D
from nDTomo.utils.noise import addpnoise1D, addpnoise3D, addpnoise2D
from nDTomo.utils.misc import cirmask
from nDTomo.ct.astra_tomo import astra_rec_vol

#%%

fn = 'particles_small.h5'

with h5py.File(fn, 'r') as f:
    
    vol = f['vol'][:]

    Amap_small = f['A'][:]
    xomap_small = f['x'][:]
    stdmap_small = f['FWHM'][:]
    slopemap_small = f['slope'][:]
    intermap_small = f['intercept'][:]
    
print(vol.shape)
print(Amap_small.shape)

#%% Now the code for the fitting

def gaussian_peaks(x, prms):
    return prms[0::3] * torch.exp(-(x - prms[1::3])**2 / (2 * prms[2::3]**2))


def gaussian_peaks_bkg(x, prms):
    return prms[0::5] * torch.exp(-(x - prms[1::5])**2 / (2 * prms[2::5]**2)) + prms[3::5]*x + prms[4::5]

im = np.sum(vol, axis = 2)

npeaks = im.shape[0]*im.shape[1]
x = np.arange(0, 10, 0.1, dtype='float32')

xv = torch.tensor(x, dtype=torch.float32, device=device)

simpats = torch.tensor(vol, dtype=torch.float32, device=device)
simpats = torch.reshape(simpats, (simpats.shape[0]*simpats.shape[1], simpats.shape[2]))

MSE = nn.MSELoss()
MAE = nn.L1Loss()

#%

microct = np.copy(im)
microct[microct>0] = 1
microct = np.reshape(microct, (microct.shape[0]*microct.shape[1], 1))
sfmap = torch.tensor(microct, requires_grad=False, device=device, dtype=torch.float32)

print(sfmap.shape)

#%%

# Perform peak fitting using the Adam optimizer
Ai = 1*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
mi = 5*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
si = 0.5*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
sl = 0.001*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
it = 0.001*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)

lower_bound = torch.zeros((npeaks*5, ), device=device, requires_grad=False)
lower_bound[0::5] = 0.1*torch.ones((npeaks,), requires_grad=False, device=device)
lower_bound[1::5] = 2*torch.ones((npeaks,), requires_grad=False, device=device)
lower_bound[2::5] = 0.05*torch.ones((npeaks,), requires_grad=False, device=device)
lower_bound[3::5] = 0*torch.ones((npeaks,), requires_grad=False, device=device)
lower_bound[4::5] = 0*torch.ones((npeaks,), requires_grad=False, device=device)

upper_bound = torch.zeros((npeaks*5, ), device=device, requires_grad=False)
upper_bound[0::5] = 5*torch.ones((npeaks,), requires_grad=False, device=device)
upper_bound[1::5] = 8*torch.ones((npeaks,), requires_grad=False, device=device)
upper_bound[2::5] = 50*torch.ones((npeaks,), requires_grad=False, device=device)
upper_bound[3::5] = 0.5*torch.ones((npeaks,), requires_grad=False, device=device)
upper_bound[4::5] = 0.5*torch.ones((npeaks,), requires_grad=False, device=device)

t = torch.zeros((npeaks*5, ), device=device, dtype=torch.float32)
t[0::5] = Ai
t[1::5] = mi
t[2::5] = si
t[3::5] = sl
t[4::5] = it

t = torch.tensor(t, requires_grad=True, device=device)

#%

learning_rate = 0.1
epochs = 5000
optimizer = optim.Adam([t], lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1E-3, verbose=1)

tc = torch.reshape(t, (t.shape[0],1))

start = time.time()

for epoch in tqdm(epochs):
    
    peak_pred = gaussian_peaks_bkg(xv, tc)*sfmap
    
    # loss = MSE(peak_pred,simpats)
    loss = MAE(peak_pred,simpats)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        t[:] = t.clamp(lower_bound, upper_bound)    
    
    scheduler.step(loss)

    if optimizer.param_groups[0]['lr'] == scheduler.min_lrs[0]:
        print("Minimum learning rate reached, stopping the optimization")
        print(epoch)
        break
    
print(time.time() - start)


#%%

peak_preds = gaussian_peaks_bkg(xv, tc)*sfmap


peak_preds = peak_preds.cpu()
peak_preds = peak_preds.detach().numpy()
peak_preds = np.reshape(peak_preds, (vol.shape[0], vol.shape[1], vol.shape[2]))
print(peak_preds.shape)

hs = HyperSliceExplorer(np.concatenate((vol, peak_preds), axis = 1))
hs.explore(cmap='gray')


#%% Results

Amap = tc[0::5,0] * sfmap[:,0]
Amap = torch.reshape(Amap, (1,im.shape[0],im.shape[0]))
Amap = Amap.cpu()
Amap = Amap.detach().numpy()
imc = np.concatenate((np.transpose(Amap[0,:,:]), Amap_small), axis = 1)

plt.figure(1);plt.clf()
plt.imshow(imc, cmap = 'gray')
plt.title('Gaussian peak: Area')
plt.colorbar()
plt.show()

xmap = tc[1::5,0] * sfmap[:,0]
xmap = torch.reshape(xmap, (1,im.shape[0],im.shape[0]))
xmap = xmap.cpu()
xmap = xmap.detach().numpy()
imc = np.concatenate((np.transpose(xmap[0,:,:]), xomap_small), axis = 1)

plt.figure(2);plt.clf()
plt.imshow(imc, cmap = 'gray')
plt.title('Gaussian peak: Position')
plt.colorbar()
plt.show()


smap = tc[2::5,0] * sfmap[:,0]
smap = torch.reshape(smap, (1,im.shape[0],im.shape[0]))
smap = smap.cpu()
smap = smap.detach().numpy()
imc = np.concatenate((np.transpose(smap[0,:,:]), stdmap_small), axis = 1)

plt.figure(3);plt.clf()
plt.imshow(imc, cmap = 'gray')
plt.title('Gaussian peak: FWHM')
plt.colorbar()
plt.show()


slmap = tc[3::5,0] * sfmap[:,0]
slmap = torch.reshape(slmap, (1,im.shape[0],im.shape[0]))
slmap = slmap.cpu()
slmap = slmap.detach().numpy()
imc = np.concatenate((np.transpose(slmap[0,:,:]), slopemap_small), axis = 1)

plt.figure(4);plt.clf()
plt.imshow(imc, cmap = 'gray')
plt.title('Slope')
plt.colorbar()
plt.show()

inmap = tc[4::5,0] * sfmap[:,0]
inmap = torch.reshape(inmap, (1,im.shape[0],im.shape[0]))
inmap = inmap.cpu()
inmap = inmap.detach().numpy()
imc = np.concatenate((np.transpose(inmap[0,:,:]), intermap_small), axis = 1)

plt.figure(5);plt.clf()
plt.imshow(imc, cmap = 'gray')
plt.title('Intercept')
plt.colorbar()
plt.show()

#%%


