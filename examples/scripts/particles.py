# -*- coding: utf-8 -*-
"""
Fitting chemical imaging data

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

fn = 'C:\\Users\\Antony\\Documents\\GitHub\\nDTomo\\examples\\volumes\\particles_small.h5'

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

#%%

im = np.sum(vol, axis = 2)

npeaks = im.shape[0]*im.shape[1]
x = np.arange(0, 10, 0.1, dtype='float32')

xv = torch.tensor(x, dtype=torch.float32, device=device)

simpats = torch.tensor(vol, dtype=torch.float32, device=device)
simpats = torch.reshape(simpats, (simpats.shape[0]*simpats.shape[1], simpats.shape[2]))

MSE = nn.MSELoss()
MAE = nn.L1Loss()

#%% Prepare a mask for the void

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

for epoch in tqdm(range(epochs)):
    
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
hs.explore(cmap='jet')


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







#%% Now let's try with a CNN

def gaussian_peaks_bkg_t(x, A, xo, st, sl, i):
    return A * torch.exp(-(x - xo)**2 / (2 * st**2)) + sl*x + i


class CNN2D(nn.Module):
    
    def __init__(self, npix, nch=5, nfilts=32, nlayers =4):
        
        super(CNN2D, self).__init__()
        
        
        conv2d_in = nn.Conv2d(nch, nfilts, kernel_size=3, stride=1, padding='same')

        conv2d_layer = nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same')

        relu =  nn.ReLU()

        conv2d_final = nn.Conv2d(nfilts, nch, kernel_size=3, stride=1, padding='same')
           
        layers = []
        layers.append(conv2d_in)
        layers.append(relu)
        
        for layer in range(nlayers):
            layers.append(conv2d_layer)
            layers.append(conv2d_layer)
            layers.append(relu)

        layers.append(conv2d_final)
        
        self.cnn2d = nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.cnn2d(x) #+ x
        
        return(out)


model = CNN2D(npix = 256, nlayers=1, nch=5, nfilts=32).to(device)

print(model)

#%%

microct = np.copy(im)
microct[microct>0] = 1
microct = np.reshape(microct, (microct.shape[0]*microct.shape[1], 1))
sfmap = torch.tensor(microct, requires_grad=False, device=device, dtype=torch.float32)

#%%

npix = 256
nch = vol.shape[2]

learning_rate = 0.001
epochs = 10000
min_lr=1E-5

# Perform peak fitting using the Adam optimizer
Ai = 1*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
mi = 5*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
si = 0.5*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
sl = 0.001*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
it = 0.001*torch.ones((npeaks,), requires_grad=True, device=device, dtype=torch.float32)
prms = torch.concat((Ai, mi, si, sl, it), dim=0).to(device)
print(prms.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, min_lr=min_lr, verbose=1)

for epoch in tqdm(range(epochs)):

    # t = torch.zeros((npeaks*5, ), device=device, dtype=torch.float32)
    
    gen_im = model(prms)
    # gen_im = torch.abs(gen_im)
    
    gen_im = torch.reshape(gen_im, (gen_im.shape[0], gen_im.shape[1]*gen_im.shape[2]))
    gen_im = torch.transpose(gen_im, 1,0)


    #### Approach 1 ####
    # t[0::5] = gen_im[:,0]
    # t[1::5] = gen_im[:,1]
    # t[2::5] = gen_im[:,2]
    # t[3::5] = gen_im[:,3]
    # t[4::5] = gen_im[:,4]

    # tc = torch.reshape(t, (t.shape[0],1))
    
    # peak_pred = gaussian_peaks_bkg(xv, tc)*sfmap
    # peak_pred = torch.reshape(peak_pred, (npix, npix, nch))
    # peak_pred = torch.transpose(peak_pred, 1, 0)
    # peak_pred = torch.reshape(peak_pred, (npix*npix, nch))
    
    #### Approach 2 ####
    peak_pred = gaussian_peaks_bkg_t(xv, gen_im[:,0:1], gen_im[:,1:2], gen_im[:,2:3], gen_im[:,3:4], gen_im[:,4:5])*sfmap
    
    loss = MAE(simpats, peak_pred)
    
    if epoch % 10 == 0:
        print(loss)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
            
    scheduler.step(loss)

    if optimizer.param_groups[0]['lr'] == scheduler.min_lrs[0]:
        print("Minimum learning rate reached, stopping the optimization")
        print(epoch)
        break

#%%

gen_im = model(prms)
gen_im = torch.abs(gen_im)

gen_im = torch.reshape(gen_im, (gen_im.shape[0], gen_im.shape[1]*gen_im.shape[2]))
gen_im = torch.transpose(gen_im, 1,0)


t = torch.zeros((npeaks*5, ), device=device, dtype=torch.float32)
t[0::5] = gen_im[:,0]
t[1::5] = gen_im[:,1]
t[2::5] = gen_im[:,2]
t[3::5] = gen_im[:,3]
t[4::5] = gen_im[:,4]

tc = torch.reshape(t, (t.shape[0],1))

peak_pred = gaussian_peaks_bkg(xv, tc)*sfmap

peak_pred = peak_pred.cpu()
peak_pred = peak_pred.detach().numpy()
peak_pred = np.reshape(peak_pred, (vol.shape[0], vol.shape[1], vol.shape[2]))

peak_pred[peak_pred<0] = 0

print(peak_pred.shape)

hs = HyperSliceExplorer(np.concatenate((vol, peak_pred), axis = 1))
hs.explore(cmap='gray')



#%% Let's do a test using the ground truth maps  - this is to check that the sfmap and the orientation of the matrices are correct

microct = np.transpose(np.copy(im))
microct[microct>0] = 1
microct = np.reshape(microct, (microct.shape[0]*microct.shape[1], 1))
sfmap = torch.tensor(microct, requires_grad=False, device=device, dtype=torch.float32)

t = torch.zeros((npeaks*5, ), dtype=torch.float32)

Amap_small_t = torch.tensor(Amap_small)
xomap_small_t = torch.tensor(xomap_small)
stdmap_small_t = torch.tensor(stdmap_small)
slopemap_small_t = torch.tensor(slopemap_small)
intermap_small_t = torch.tensor(intermap_small)

Amap_small_t = torch.reshape(Amap_small_t, (1,Amap_small_t.shape[0],Amap_small_t.shape[1]))
xomap_small_t = torch.reshape(xomap_small_t, (1,xomap_small_t.shape[0],xomap_small_t.shape[1]))
stdmap_small_t = torch.reshape(stdmap_small_t, (1,stdmap_small_t.shape[0],stdmap_small_t.shape[1]))
slopemap_small_t = torch.reshape(slopemap_small_t, (1,slopemap_small_t.shape[0],slopemap_small_t.shape[1]))
intermap_small_t = torch.reshape(intermap_small_t, (1,intermap_small_t.shape[0],intermap_small_t.shape[1]))


gen_im = torch.concat((Amap_small_t, xomap_small_t, stdmap_small_t, slopemap_small_t, intermap_small_t), dim=0)
gen_im = torch.reshape(gen_im, (gen_im.shape[0], gen_im.shape[1]*gen_im.shape[2]))
gen_im = torch.transpose(gen_im, 1,0)

t[0::5] = gen_im[:,0]
t[1::5] = gen_im[:,1]
t[2::5] = gen_im[:,2]
t[3::5] = gen_im[:,3]
t[4::5] = gen_im[:,4]

tc = torch.reshape(t, (t.shape[0],1)).to(device)

peak_pred = gaussian_peaks_bkg(xv, tc)*sfmap

peak_pred = peak_pred.cpu()
peak_pred = peak_pred.detach().numpy()
peak_pred = np.reshape(peak_pred, (vol.shape[0], vol.shape[1], vol.shape[2]))
peak_pred = np.transpose(peak_pred, (1,0,2))
peak_pred[peak_pred<0] = 0

print(peak_pred.shape)

pats = torch.reshape(simpats, (npix, npix, nch))
pats = pats.cpu()
pats = pats.detach().numpy()

hs = HyperSliceExplorer(np.concatenate((pats, peak_pred), axis = 1))
hs.explore(cmap='gray')












