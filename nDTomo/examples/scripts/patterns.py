# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:26:50 2021

@author: Antony
"""

from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares
import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob, time, h5py
from pathlib import Path
import hyperspy.api as hs
from skimage.transform import iradon, radon
from pathlib import Path

import nDTomo.sim.xrd1d.Phases as ph

#%%

CIFlist = [f for f in glob.glob("*.cif")]

print(CIFlist)

ucpars = np.zeros((len(CIFlist),6))

phasesinfo = {}
phasesinfo['SpaceGroupNumber'] = {}
phasesinfo['CrystalClass'] = {}
phasesinfo['Formula'] = {}

phase = {}
sgn = []
for ii in range(len(CIFlist)):
    phase[ii] = ph.phase()
    
    try:
        phase[ii].readCIFinfo(CIFlist[ii], dispinfo = False)
    
        ucpars[ii,:] = phase[ii].ucellpars
        phasesinfo['SpaceGroupNumber'][ii] = phase[ii].sgno
        phasesinfo['CrystalClass'][ii] = phase[ii].crystalClass
        phasesinfo['Formula'][ii] = phase[ii].cform
        sgn.append(phase[ii].sgno)
    
    except:
        print('Failed to read CIF ', CIFlist[ii])
        
        
#%%

sgn_array = np.array(sgn)

i, indi, ind, unique_counts = np.unique(sgn_array, return_index
                                        = True, return_inverse=True, return_counts=True)

for ii in range(len(i)):
    print('Space group: ',i[ii],' Crystal system', phasesinfo['CrystalClass'][indi[ii]], ' Appearance times: ', unique_counts[ii], ' Result group: ', ii,' Formula', phasesinfo['Formula'][indi[ii]])


#%%

q = np.arange(0.1, 10, 0.01)
E = 100 # keV
wvl = 12.398/E
d = 2 * np.pi/ q;
tth = np.rad2deg(2*np.arcsin(wvl/(2*d)));

print(tth.shape)

#%%

plt.figure(1);plt.clf()

dps = []

for ii in range(len(phase)):

    start=time.time()
    
    phase[ii].set_wvl(wvl)
    
    
    phase[ii].symmetry(sintlmax = 0.35)
    
    phase[ii].instpars(U= 0.24902095, V= -0.012046, W= 0.00046532)
    
    
    phase[ii].set_xaxis(tth_1d = tth)
    
    
    phase[ii].ph_cls(CLS = 10)
    
    
    I = phase[ii].create_pattern()
        
    I = I/np.max(I)

    dps.append(I)
    
    
    plt.plot(phase[ii].tth_1d, I*0.5 - ii*0.1 - 0.1);plt.show();
    
    print(phasesinfo['Formula'][ii])

dps = np.array(dps)

plt.figure(2);plt.clf()
plt.plot(np.sum(dps, axis = 0))

#%%

roi = np.arange(200, 450)

plt.figure(2);plt.clf()
plt.plot(np.sum(dps[:,roi], axis = 0))

dpAl = dps[0,roi] + 0.01
dpCu = dps[1,roi] + 0.01
dpFe = dps[2,roi] + 0.01
dpPt = dps[3,roi] + 0.01
dpZn = dps[4,roi] + 0.01

plt.figure(3);plt.clf()
plt.plot(dpAl)

#%% Export the patterns

fn = 'Patterns.h5'

h5fn = h5py.File(fn, "w")
h5fn.create_dataset('Al', data = dpAl)
h5fn.create_dataset('Cu', data = dpCu)
h5fn.create_dataset('Fe', data = dpFe)
h5fn.create_dataset('Pt', data = dpPt)
h5fn.create_dataset('Zn', data = dpZn)
h5fn.create_dataset('q', data = q[roi])
h5fn.create_dataset('tth', data = tth[roi])
h5fn.close()