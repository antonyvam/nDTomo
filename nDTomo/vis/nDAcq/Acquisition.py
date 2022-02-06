# -*- coding: utf-8 -*-
"""
nDAacq to simulate an XRD-CT dataset with 2D diffraction patterns from 1D diffraction patterns

@author: Antony Vamvakeros
"""

#%%

from nDTomo.utils.misc import ndtomopath, h5read_data, h5write_data, closefigs, showplot, showspectra, showim, addpnoise1D, addpnoise2D, KeVtoAng
from nDTomo.sim.shapes.phantoms import load_example_patterns, nDphantom_2D, nDphantom_3D
from nDTomo.ct.astra_tomo import astra_create_geo, astre_rec_vol, astre_rec_alg, astra_create_sino_geo, astra_create_sino

from tqdm import tqdm
import time, fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
from pyFAI.calibrant import get_calibrant
import numpy as np

#%%


class nDAacq():
    
    '''
    nDAacq simulates the acquisition of XRD-CT data using sinogram data
    '''
    
    def __init__(self, mask = None, file_format ='cbf'):
                        
        self.mask = mask; self.file_format = file_format
        
    def setdata(self, data, xaxis, units = "q_A^-1", scantype = 'Zig', fastaxis = 'Translation', slowaxis = 'Rotation'):
        
        '''
        Data contains the 3D array corresponding to the sinogram data (ntrans, nproj, nch)
        xaxis is the xaxis for the 1D pattens
        '''        
        self.data = data; self.xaxis = xaxis
        self.ntrans = self.data.shape[0]; self.nproj = self.data.shape[1]; 
        self.units = units; self.scantype = scantype
        self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    def setdirs(self, savedir, dname):
        
        self.savedir = savedir; self.dname = dname
        
    def run(self):
        
        self.start_scan()
    
    def start_scan(self, addnoise = 'No', ct = 0.1):
        
        '''
        Performs the data acquisition
        '''
        
        if self.scantype == 'Zig':
        
            if self.fastaxis == 'Translation' and self.slowaxis == 'Rotation':
            
                for pp in tqdm(range(self.nproj)):
                    
                    for tt in range(self.ntrans):
                    
                        self.frame = self.conv1Dto2D(self.data[tt,pp,:], msk = self.mask, units = self.units)
                    
                        if addnoise == 'Yes':
                            
                            self.frame = addpnoise2D(self.frame, ct)
                    
                        fn = '%s\\%s_%d_%d' %(self.savedir, self.dname, pp + 1, tt + 1)
                    
                        self.write_frame(fn)
                        
            elif self.fastaxis == 'Rotation' and self.slowaxis == 'Translation':
            
                for tt in tqdm(range(self.ntrans)):
                    
                    for pp in range(self.nproj):
                    
                        self.frame = self.conv1Dto2D(self.sinograms[tt,pp,:], msk = self.mask, units = self.units)
                    
                        if addnoise == 'Yes':
                            
                            self.frame = addpnoise2D(self.frame, ct)
                            
                        fn = '%s\\%s_%d_%d' %(self.savedir, self.dname, tt + 1, pp + 1)

                        self.write_frame(fn)
                    
    
    def write_frame(self, fn):
        
        if self.file_format == 'edf':
            
            self.write_edf('%s.edf' %fn)
    
        elif self.file_format == 'cbf':
            
            self.write_cbf('%s.cbf' %fn)
            
        elif self.file_format == 'tiff' or self.file_format == 'tif':
            
            self.write_tiff('%s.tiff' %fn)            
            
    def write_edf(self, fn):
        
        fabio.edfimage.EdfImage(self.frame).write(fn)
    
    def write_cbf(self, fn):
        
        fabio.cbfimage.CbfImage(self.frame).write(fn)
    
    def write_tiff(self, fn):
        
        fabio.tifimage.TifImage(self.frame).write(fn)
    
    def write_h5(self):
        
        pass
    
    def readscanprms(self):
        
        pass
    
    def setscanprms(self, trans, nproj, scantype = 'Zig', fastaxis = 'Translation', slowaxis = 'Rotation'):
        
        '''
        Scan type: Zig, Zigzag, Interlaced, ContRot
        '''
        
        self.ntrans = trans; self.nproj = nproj; 
        self.scantype = scantype; self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    
    def readponi(self, poni):
        
        ai = pyFAI.load(poni)
        self.dist = ai.dist; self.poni1 = ai.poni1; self.poni2 = ai.poni2
        self.rot1 = ai.rot1; self.rot2 = ai.rot2; self.rot3 = ai.rot3        
        self.wvl = ai.wavelength
        self.ImageD11geo = ai.getImageD11()
    
    def setwvl(self, wvl):
        
        self.wvl = wvl
    
    def setxrdprms(self, dist, poni1, poni2, rot1, rot2, rot3):
        
        self.dist = dist; self.poni1 = poni1; self.poni2 = poni2
        self.rot1 = rot1; self.rot2 = rot2; self.rot3 = rot3
        
    def calcbeamcentre(self):
        
        self.c1 = self.poni1/self.detector.pixel1
        self.c2 = self.poni2/self.detector.pixel2
    
    def setdetector(self, detpixsz1= 0.000172, detpixsz2= 0.000172, shape= (1679,1475) ):
        
        self.detector = pyFAI.detectors.Detector(pixel1=detpixsz1, pixel2=detpixsz2, max_shape=shape)
    
    def azimint(self):
        
        self.ai = AzimuthalIntegrator(dist=self.dist, poni1=self.poni1, poni2=self.poni2, rot1=self.rot1, 
                                 rot2=self.rot2, rot3=self.rot3, detector=self.detector, wavelength=self.wvl)
    
    def conv1Dto2D(self, I, msk=None, units="q_A^-1"):
        
        
        img_theo = self.ai.calcfrom1d(self.xaxis, I, mask=msk, dim1_unit=units,
                                 correctSolidAngle=False,
                                 polarization_factor=0.95)
        
        return(img_theo)

    def create_nDTomo_phantom(self, npix = 100, nproj = 110):
    
        dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
        spectra = [dpAl, dpCu, dpFe, dpPt, dpZn]
        
        self.xaxis = q
        
        iml = nDphantom_2D(npix, nim = 'Multiple')
        
        imAl, imCu, imFe, imPt, imZn = iml
        
        chemct = nDphantom_3D(npix, use_spectra = 'Yes', spectra = spectra, imgs = iml, indices = 'All',  norm = 'No')    
    
        self.data = np.zeros((chemct.shape[0], nproj, chemct.shape[2]))
        
        for ii in tqdm(range(chemct.shape[2])):
            
            proj_id = astra_create_sino_geo(chemct[:,:,ii], theta=np.deg2rad(np.arange(0, 180, 180/nproj)))
            self.data[:,:,ii] = astra_create_sino(chemct[:,:,ii], proj_id).transpose()
                
#%% Perform a test

p = ndtomopath()
poni = "%sexamples\\xrd2D\\CeO2.poni" %p
savedir = 'Y:\\Antony\\nDTomo\\test'

Acq = nDAacq()

Acq.readponi(poni)
Acq.setdetector()
Acq.azimint()
Acq.create_nDTomo_phantom(npix = 100, nproj = 110)
Acq.savedir = savedir
Acq.dname = 'xrdct_test'

#%%

im = Acq.conv1Dto2D( Acq.data[50,50,:])
im = addpnoise2D(im, ct = 0.01)

showim(im,1,cmap='gray')

#%% Perform the XRD-CT scan

Acq.start_scan(addnoise = 'Yes', ct = 0.01)



















