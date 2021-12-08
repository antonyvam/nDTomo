# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:44:31 2021

@author: Antony
"""

from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares
import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob, time, h5py
import hyperspy.api as hs
from skimage.transform import iradon, radon
import h5py
from nDTomo.utils.misc import ndtomopath


'''
Need to convert the function to class
'''


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
    

def phantom5c(npix):
        
    im1 = sstar(npix, nstars=32)
    im2 = dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2)
    im3 = ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05)
    im4 = tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2])
    im5 = face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 )
    
    return(im1, im2, im3, im4, im5)


    
    

def load_patterns():
    

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
























