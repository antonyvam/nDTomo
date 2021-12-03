# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:55:36 2019

@author: Antony
"""

import numpy as np
import h5py

tand = lambda x: np.tan(x*np.pi/180.)
atand = lambda x: 180.*np.arctan(x)/np.pi

def Linear(x, a=0, b=0):
	return a*x+b

def Quadratic(x, a=0, b=0, c=0):
	return a*x**2 + b*x + c

def Gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    return (2.0*np.sqrt(np.log(2.0)/np.pi))*(amplitude/sigma ) * np.exp(-(4.0*np.log(2.0)/sigma**2) * (x - center)**2)
    
def Lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    return (2.0 / np.pi) * (amplitude / sigma) /  (1 + (4.0 / sigma**2) * (x - center**2))


def G(x, center=0.0, sigma=1.0):
    return (2.0*np.sqrt(np.log(2.0)/np.pi))*(1/sigma ) * np.exp(-(4.0*np.log(2.0)/sigma**2) * (x - center)**2)    
def L(x, center=0.0, sigma=1.0):
    return (2.0 / np.pi) * (1 / sigma) /  (1 + (4.0 / sigma**2) * (x - center**2))
def PVoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    return amplitude * ((1-fraction) * G(x, center, sigma) + fraction * L(x, center, sigma))

def CombGaussians(x, pars):
    return (Linear(x, pars[-2], pars[-1]) + np.sum(Gaussian(np.transpose(np.tile(x, ((int((len(pars)-2)/3),1)))), amplitude=pars[0:int((len(pars)-2)/3)], center=pars[int((len(pars)-2)/3):2*int((len(pars)-2)/3)], sigma=pars[2*int((len(pars)-2)/3):3*int((len(pars)-2)/3)]), axis = 1))

def CombGaussiansLS(pars, x, y):
    return((Linear(x[:,0], pars[-2], pars[-1]) + np.sum(Gaussian(x, amplitude=pars[0:int((len(pars)-2)/3)], center=pars[int((len(pars)-2)/3):2*int((len(pars)-2)/3)], sigma=pars[2*int((len(pars)-2)/3):3*int((len(pars)-2)/3)]), axis = 1))-y)

def CombGaussiansq(x, pars):
    return (Quadratic(x, pars[-3], pars[-2], pars[-1]) + np.sum(Gaussian(np.transpose(np.tile(x, ((int((len(pars)-2)/3),1)))), amplitude=pars[0:int((len(pars)-2)/3)], center=pars[int((len(pars)-2)/3):2*int((len(pars)-2)/3)], sigma=pars[2*int((len(pars)-2)/3):3*int((len(pars)-2)/3)]), axis = 1))

def CombGaussiansLSq(pars, x, y):
    return((Quadratic(x[:,0], pars[-3], pars[-2], pars[-1]) + np.sum(Gaussian(x, amplitude=pars[0:int((len(pars)-2)/3)], center=pars[int((len(pars)-2)/3):2*int((len(pars)-2)/3)], sigma=pars[2*int((len(pars)-2)/3):3*int((len(pars)-2)/3)]), axis = 1))-y)

def GetPeaksInfo(res):

    Area = res[0:int((len(res)-2)/3)]    
    Pos = res[int((len(res)-2)/3):2*int((len(res)-2)/3)]    
    FWHM = res[2*int((len(res)-2)/3):3*int((len(res)-2)/3)]
    return(Area, Pos, FWHM)
    
def Cagliotti(twotheta, U, V, W):
    # return np.sqrt(U*tantth**2 + V*tantth + W)
    return(np.sqrt(U*tand(twotheta / 2.0)**2 + V*tand(twotheta / 2.0) + W))

def CagliottiLS(pars, tantth, y):
    return y - (Cagliotti(tantth[:,0], pars[0], pars[1], pars[2]))



def savepeakfits(fn, Areas, Pos, Sigma):
    '''
    It takes a filename (use .h5 or .hdf5) and the Areas, Pos, Sigma which are dictionaries
    '''
    
    h5f = h5py.File(fn, "w")
    
    for ii in range(0,len(Areas.keys())):
        
        h5f.create_dataset('Peak_Area_%d' %ii, data=Areas[ii])
        h5f.create_dataset('Peak_Position_%d' %ii, data=Pos[ii])
        h5f.create_dataset('Peak_FWHM_%d' %ii, data=Sigma[ii])
        
    h5f.close()
	
def loadpeakfits(fn, peaks):
    '''
    It takes a filename (use .h5 or .hdf5) and the Areas, Pos, Sigma which are dictionaries
    '''

    Areas = {}
    Pos = {}
    Sigma = {}
    
    with h5py.File(fn,'r') as f:
        
        for ii in range(0,peaks):
            
            Areas[ii] = f['/Peak_Area_%d' %ii][:]
            Pos[ii] = f['/Peak_Position_%d' %ii][:]
            Sigma[ii] = f['/Peak_FWHM_%d' %ii][:]
            
        Bkga = f['/Bkga'][:]
        Bkgb = f['/Bkgb'][:]
            
        try:
            Bkgc = f['/Bkgc'][:]
        except:
            pass
        
    return Areas, Pos, Sigma, Bkga, Bkgb, Bkgc