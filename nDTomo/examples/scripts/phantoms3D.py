# -*- coding: utf-8 -*-
"""
Creating a 3D phantoms for testing neural networks

@author: Antony Vamvakeros
"""

#%%

from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares
import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob, time, h5py
import hyperspy.api as hs
from skimage.transform import iradon, radon


#%% Create the phase maps

'''
Let's create five maps that correspond to five sample components
'''

from nDTomo.sim.shapes.phantoms import phantom5c

nt = 256

imAl, imCu, imFe, imPt, imZn = phantom5c(nt)

plt.figure(1);plt.clf();
plt.imshow(imAl, cmap = 'jet')
plt.colorbar()
plt.show()


plt.figure(2);plt.clf();
plt.imshow(imCu, cmap = 'jet')
plt.colorbar()
plt.show()


plt.figure(3);plt.clf();
plt.imshow(imFe, cmap = 'jet')
plt.colorbar()
plt.show()


plt.figure(4);plt.clf();
plt.imshow(imPt, cmap = 'jet')
plt.colorbar()
plt.show()


plt.figure(5);plt.clf();
plt.imshow(imZn, cmap = 'jet')
plt.colorbar()
plt.show()

#%% Read the patterns

'''
Let's read the diffraction patterns that correspond to the five components
'''

with h5py.File('nDTomo/examples/patterns/patterns.h5', 'r') as f:
    
    print(f.keys())

    dpAl = np.array(f['Al'][:])
    dpCu = np.array(f['Cu'][:])
    dpFe = np.array(f['Fe'][:])
    dpPt = np.array(f['Pt'][:])
    dpZn = np.array(f['Zn'][:])

    tth = np.array(f['tth'][:])
    q = np.array(f['q'][:])

print(dpAl.shape, tth.shape, tth.shape)

#%% Plotting

'''
Let's plot the five diffraction patterns
'''

plt.figure(1);plt.clf()

plt.plot(q, dpAl)
plt.plot(q, dpCu+0.05*1)
plt.plot(q, dpFe+0.05*2)
plt.plot(q, dpPt+0.05*3)
plt.plot(q, dpZn+0.05*4)


#%% Phantoms

'''
Create the XRD map and micro-CT phantoms
'''

xrdct_Al = np.tile(dpAl, (nt, nt, 1))
xrdct_Cu = np.tile(dpCu, (nt, nt, 1))
xrdct_Fe = np.tile(dpFe, (nt, nt, 1))
xrdct_Pt = np.tile(dpPt, (nt, nt, 1))
xrdct_Zn = np.tile(dpZn, (nt, nt, 1))

for ii in range(xrdct_Al.shape[2]):
    
    xrdct_Al[:,:,ii] = xrdct_Al[:,:,ii]*imAl
    xrdct_Cu[:,:,ii] = xrdct_Cu[:,:,ii]*imCu
    xrdct_Fe[:,:,ii] = xrdct_Fe[:,:,ii]*imFe
    xrdct_Pt[:,:,ii] = xrdct_Pt[:,:,ii]*imPt
    xrdct_Zn[:,:,ii] = xrdct_Zn[:,:,ii]*imZn


xrdct = xrdct_Al + xrdct_Cu + xrdct_Fe + xrdct_Pt + xrdct_Zn

print(xrdct.shape)

#%% Create the sinogram XRD-CT volume and add Poisson noise

nproj = 2*nt
theta = np.arange(0, 180, 180/nproj)

s = np.zeros((xrdct.shape[0], nproj, xrdct.shape[2]))

for ii in range(xrdct.shape[2]):
    
    s[:,:,ii] = radon(xrdct[:,:,ii], theta)

    print(ii)
    
#%%

plt.figure(1);plt.clf()
plt.plot(np.mean(np.mean(xrdct, axis =0), axis = 0))
plt.show()

#%%

photon_count = 500

im = xrdct[:,:,53] + 1E-9
imn = np.random.poisson(im * photon_count)/ photon_count
# imn = imn * 1 # To make the reconstructed image be approaximately in the 0-1 range

plt.figure(1);plt.clf();
plt.imshow(imn, cmap = 'jet')
plt.colorbar()
plt.show()

#%%

xrdct_n1 = np.zeros_like(xrdct)
xrdct_n2 = np.zeros_like(xrdct)

for ii in range(xrdct.shape[2]):
    
    im = xrdct[:,:,ii] + 1E-9
    xrdct_n1[:,:,ii] = np.random.poisson(im * photon_count)/ photon_count
    xrdct_n2[:,:,ii] = np.random.poisson(im * photon_count)/ photon_count
    
    print(ii)


#%% Export the volumes

with h5py.File('data/Phantom_noisy.h5', "w") as f:
    f.create_dataset('ground_truth', data = xrdct)
    f.create_dataset('vol1', data = xrdct_n1)
    f.create_dataset('vol2', data = xrdct_n2)

f.close()

#%% Read the spectra

'''
Let's read the XANES spectra that correspond to the five components
'''

with h5py.File('nDTomo/examples/patterns/AllSpectra.h5', 'r') as f:
    
    print(f.keys())

    sNMC = np.array(f['NMC'][:])
    sNi2O3 = np.array(f['Ni2O3'][:])
    sNiOH2 = np.array(f['NiOH2'][:])
    sNiS = np.array(f['NiS'][:])
    sNifoil = np.array(f['Nifoil'][:])

    E = np.array(f['energy'][:])

print(sNMC.shape, E.shape)

#%% Plotting

'''
Let's plot the five diffraction patterns
'''

plt.figure(1);plt.clf()

plt.plot(E, sNMC)
plt.plot(E, sNi2O3+0.05*1)
plt.plot(E, sNiOH2+0.05*2)
plt.plot(E, sNiS+0.05*3)
plt.plot(E, sNifoil+0.05*4)

#%% Phantoms

'''
Create the XRD map and micro-CT phantoms
'''

xanesct_NMC = np.tile(sNMC, (nt, nt, 1))
xanesct_Ni2O3 = np.tile(sNi2O3, (nt, nt, 1))
xanesct_NiOH2 = np.tile(sNiOH2, (nt, nt, 1))
xanesct_NiS = np.tile(sNiS, (nt, nt, 1))
xanesct_Nifoil = np.tile(sNifoil, (nt, nt, 1))

for ii in range(xanesct_NMC.shape[2]):
    
    xanesct_NMC[:,:,ii] = xanesct_NMC[:,:,ii]*imAl
    xanesct_Ni2O3[:,:,ii] = xanesct_Ni2O3[:,:,ii]*imCu
    xanesct_NiOH2[:,:,ii] = xanesct_NiOH2[:,:,ii]*imFe
    xanesct_NiS[:,:,ii] = xanesct_NiS[:,:,ii]*imPt
    xanesct_Nifoil[:,:,ii] = xanesct_Nifoil[:,:,ii]*imZn


xanesct = xanesct_NMC + xanesct_Ni2O3 + xanesct_NiOH2 + xanesct_NiS + xanesct_Nifoil

print(xanesct.shape)

#%% Export the volumes

with h5py.File('data/Phantom_xanes.h5', "w") as f:
    f.create_dataset('data', data = xanesct)
f.close()


#%%


microct = np.zeros_like(xrdct)

for ii in range(microct.shape[2]):
    
    microct[:,:,ii] = xrdct[:,:,ii]/np.max(xrdct[:,:,ii])

    print(ii)
    
print(microct.shape)


#%% Inspect the volume

'''
We can use hyperspy to quickly inspect the 3D volumes
'''

sv = hs.signals.Signal2D(np.transpose(xrdct, (2,1,0)))
 
sv.plot()

sv = hs.signals.Signal2D(np.transpose(microct, (2,1,0)))
 
sv.plot()

#%% Radon / iradon

'''
Let's create a noisy sinogram and reconstruct
'''

ims = microct[:,:,0]

nproj = 2*nt
theta = np.arange(0, 180, 180/nproj)

photon_count = 5000

fp = radon(ims, theta)
fp = fp/np.max(fp)
fp = np.random.poisson(fp * photon_count)/ photon_count
fp = fp * 1.5E2 # To make the reconstructed image be approaximately in the 0-1 range

plt.figure(1);plt.clf();
plt.imshow(fp, cmap = 'jet')
plt.colorbar()
plt.show()

rec = iradon(fp, theta, nt)

plt.figure(2);plt.clf();
plt.imshow(rec, cmap = 'jet')
plt.colorbar()

#%% Create the sinogram XRD-CT volume and add Poisson noise

s = np.zeros((fp.shape[0], fp.shape[1], xrdct.shape[2]))

for ii in range(xrdct.shape[2]):
    
    s[:,:,ii] = radon(xrdct[:,:,ii], theta)

    print(ii)

#% Add noise

sp = s/np.max(s) + 1E-6

for ii in range(s.shape[2]):
    
    sp[:,:,ii] = np.random.poisson(sp[:,:,ii] * photon_count)/ photon_count
    
    print(ii)

#%% Create three reconstructed volumes one with all projections and two interlaced with half the projections

rec = np.zeros((nt, nt, sp.shape[2]))
rec1 = np.zeros((nt, nt, sp.shape[2]))
rec2 = np.zeros((nt, nt, sp.shape[2]))

for ii in range(sp.shape[2]):

    rec[:,:,ii] = iradon(sp[:,:,ii], theta, nt)
    rec1[:,:,ii] = iradon(sp[:,0::2,ii], theta[0::2], nt)
    rec2[:,:,ii] = iradon(sp[:,1::2,ii], theta[1::2], nt)
    
    print(ii)

#%% Export the volumes

with h5py.File('./data/Phantom_xrdct.h5', "w") as f:
    f.create_dataset('ground_truth', data = xrdct)
    f.create_dataset('vol', data = rec)
    f.create_dataset('vol1', data = rec1)
    f.create_dataset('vol2', data = rec2)
    f.create_dataset('q', data = q)
    f.create_dataset('tth', data = tth)
    f.create_dataset('sinograms', data = s)
    f.create_dataset('sinograms_noisy', data = sp)
    f.create_dataset('theta', data = theta)

f.close()


#%% Create the sinogram micro-CT volume and add Poisson noise

sm = np.zeros((fp.shape[0], fp.shape[1], microct.shape[2]))

for ii in range(microct.shape[2]):
    
    sm[:,:,ii] = radon(microct[:,:,ii], theta)

    print(ii)

#% Add noise

spm = sm/np.max(s) + 1E-6

for ii in range(sm.shape[2]):
    
    spm[:,:,ii] = np.random.poisson(spm[:,:,ii] * photon_count)/ photon_count
    
    print(ii)

#%% Create three reconstructed volumes one with all projections and two interlaced with half the projections

recm = np.zeros((nt, nt, sp.shape[2]))
recm1 = np.zeros((nt, nt, sp.shape[2]))
recm2 = np.zeros((nt, nt, sp.shape[2]))

for ii in range(spm.shape[2]):

    recm[:,:,ii] = iradon(spm[:,:,ii], theta, nt)
    recm1[:,:,ii] = iradon(spm[:,0::2,ii], theta[0::2], nt)
    recm2[:,:,ii] = iradon(spm[:,1::2,ii], theta[1::2], nt)
    
    print(ii)

#%% Export the volumes

with h5py.File('./data/Phantom_microct.h5', "w") as f:
    f.create_dataset('ground_truth', data = microct)
    f.create_dataset('vol', data = recm)
    f.create_dataset('vol1', data = recm1)
    f.create_dataset('vol2', data = recm2)
    f.create_dataset('sinograms', data = sm)
    f.create_dataset('sinograms_noisy', data = spm)
    f.create_dataset('theta', data = theta)

f.close()































