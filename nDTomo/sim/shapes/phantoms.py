# -*- coding: utf-8 -*-
"""
Create phantoms (2D/3D) for image processing and analysis experiments

@author: Antony Vamvakeros
"""

from nDTomo.utils.misc import ndtomopath
from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares
import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob, time, h5py
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
import hyperspy.api as hs

'''
Need to convert the function to class
'''

def SheppLogan(npix):

    '''
    Create a Shepp Logan phantom using skimage
    '''

    im = shepp_logan_phantom()
    im = rescale(im, scale=npix/im.shape[0], mode='reflect')
    return(im)

def sstar(npix, nstars=32):
    phase = SiemensStar(nstars)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2):
    phase = DogaCircles(n_sizes=n_sizes, size_ratio=size_ratio, n_shuffles=n_shuffles)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05):
    phase = SlantedSquares(count=count, angle=angle, gap=gap)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2]):
    
    m = Mesh()
    m.append(Triangle(Point(p1), Point(p2), Point(p3)))
    phase = Phantom(geometry=m)
    phase.material = SimpleMaterial(mass_attenuation=1.0)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 ):
    
    # Make a circle with a triangle cut out
    m = Mesh()
    m.append(Circle(Point(cp), radius=cr))
    m.append(Triangle(Point(tp1), Point(tp2), Point(tp3)))
    
    phase = Phantom(geometry=m)
    
    # Make two eyes separately
    eyeL = Phantom(geometry=Circle(Point(e1p), radius=e1r))
    eyeR = Phantom(geometry=Circle(Point(e2p), radius=e2r))
    
    phase.material = SimpleMaterial(mass_attenuation=1.0)
    eyeL.material = SimpleMaterial(mass_attenuation=1.0)
    eyeR.material = SimpleMaterial(mass_attenuation=1.0)
    
    phase.append(eyeL)
    phase.append(eyeR)
    
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    
    return(im)
    
def phantom1c(npix):
        
    im1 = sstar(npix, nstars=32)
    im2 = dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2)
    im3 = ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05)
    im4 = tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2])
    im5 = face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 )
    
    im = im1 + im2 + im3 + im4 + im5
    im = im/np.max(im)
    return(im)

def phantom5c(npix):
        
    im1 = sstar(npix, nstars=32)
    im2 = dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2)
    im3 = ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05)
    im4 = tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2])
    im5 = face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 )
    
    return(im1, im2, im3, im4, im5)

def load_example_patterns():
    
    fn = '%s\examples\patterns\patterns.h5' %(ndtomopath())

    with h5py.File(fn, 'r') as f:
        
        print(f.keys())
        
        dpAl = np.array(f['Al'][:])
        dpCu = np.array(f['Cu'][:])
        dpFe = np.array(f['Fe'][:])
        dpPt = np.array(f['Pt'][:])
        dpZn = np.array(f['Zn'][:])
        
        tth = np.array(f['tth'][:])
        q = np.array(f['q'][:])

    return(dpAl, dpCu, dpFe, dpPt, dpZn, tth, q)

def phantom5c_microct(npix):
    
    
    imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
    dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
    
    vol_Al = np.tile(dpAl, (npix, npix, 1))
    vol_Cu = np.tile(dpCu, (npix, npix, 1))
    vol_Fe = np.tile(dpFe, (npix, npix, 1))
    vol_Pt = np.tile(dpPt, (npix, npix, 1))
    vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
    microct =  np.zeros_like(vol_Al)
    
    for ii in range(vol_Al.shape[2]):
        
        im = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
              + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
        
        microct[:,:,ii] = im/np.max(im)    

    return(microct)

def phantom5c_xrdct(npix):
    
    
    imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
    dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
    
    vol_Al = np.tile(dpAl, (npix, npix, 1))
    vol_Cu = np.tile(dpCu, (npix, npix, 1))
    vol_Fe = np.tile(dpFe, (npix, npix, 1))
    vol_Pt = np.tile(dpPt, (npix, npix, 1))
    vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
    xrdct =  np.zeros_like(vol_Al)
    
    for ii in range(xrdct.shape[2]):
        
        xrdct[:,:,ii] = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
              + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
        
    return(xrdct)

def load_example_xanes():
    

    fn = '%s\examples\patterns\AllSpectra.h5' %(ndtomopath())

    with h5py.File(fn, 'r') as f:
        
        print(f.keys())
        
        sNMC = np.array(f['NMC'][:])
        sNi2O3 = np.array(f['Ni2O3'][:])
        sNiOH2 = np.array(f['NiOH2'][:])
        sNiS = np.array(f['NiS'][:])
        sNifoil = np.array(f['Nifoil'][:])
    
        E = np.array(f['energy'][:])

    return(sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E)

def phantom5c_xanesct(npix):
    
    
    imNMC, imNi2O3, imNiOH2, imNiS, imNifoil = phantom5c(npix)
    sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E = load_example_xanes()
    
    vol_NMC = np.tile(sNMC, (npix, npix, 1))
    vol_Ni2O3 = np.tile(sNi2O3, (npix, npix, 1))
    vol_NiOH2 = np.tile(sNiOH2, (npix, npix, 1))
    vol_NiS = np.tile(sNiS, (npix, npix, 1))
    vol_Nifoil = np.tile(sNifoil, (npix, npix, 1))
    
    xanesct =  np.zeros_like(vol_NMC)
    
    for ii in range(xanesct.shape[2]):
        
        xanesct[:,:,ii] = (vol_NMC[:,:,ii]*imNMC +  vol_Ni2O3[:,:,ii]*imNi2O3 + vol_NiOH2[:,:,ii]*imNiOH2
              + vol_NiS[:,:,ii]*imNiS + vol_Nifoil[:,:,ii]*imNifoil)
        
    return(xanesct)










































