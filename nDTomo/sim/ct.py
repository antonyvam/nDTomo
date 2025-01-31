# -*- coding: utf-8 -*-

"""
nDAacq to simulate an XRD-CT dataset with 2D diffraction patterns from 1D diffraction patterns

@author: Antony Vamvakeros
"""

#%%

from nDTomo.utils.noise import addpnoise2D
from nDTomo.sim.shapes.phantoms import load_example_patterns, nDphantom_2D, nDphantom_3D
from nDTomo.ct.astra_tomo import astra_create_sino

from tqdm import tqdm
import time, fabio, h5py
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
from pyFAI.calibrant import get_calibrant
import numpy as np

class nDVAcq():
    
    '''
    nDVAcq simulates the acquisition of 2D X-ray scatter tomography data using 1D X-ray scatter tomography sinogram data
    '''
    
    def __init__(self, mask = None, file_format ='cbf', scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1"):
                        
        self.mask = mask; self.file_format = file_format; self.units = units;
        self.scantype = scantype
        self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    def setdata(self, data, xaxis):
        
        '''
        Data contains the 3D array corresponding to the sinogram data (ntrans, nproj, nch)
        xaxis is the xaxis for the 1D pattens
        '''        
        self.data = data; self.xaxis = xaxis
        self.ntrans = self.data.shape[0]; self.nproj = self.data.shape[1]; 
        
        
    def setdirs(self, savedir, dname):
        
        self.savedir = savedir; self.dname = dname
        
    def run(self):
        
        self.start_scan()
    
    def start_scan(self, addnoise = 'No', ct = 0.1):
        
        '''
        Performs the data acquisition
        '''
        
        print('Starting the virtual scan')
        print('The acquisition strategy is: %s' %(self.scantype))
        print('Fast axis is %s, slow axis is %s' %(self.fastaxis, self.slowaxis))
        print('The raw data will be saved as %s' %(self.file_format))
        
        if self.scantype == 'Zigzig':
        
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
                        
        
        print('Exporting the scan parameters')
        self.saveprms()
                    
    
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
    
    def readscanprms(self, fn=None):
        
        if fn is None:
            
            fn = '%s\\%s_scan_prms.h5' %(self.savedir, self.dname)
        
        with h5py.File(fn, 'r') as f:
    
            print(f.keys())        
    
            self.ntrans = np.array(f['ntrans'])
            self.nproj = np.array(f['nproj'])
            self.fastaxis = np.array(f['fastaxis'])
            self.slowaxis = np.array(f['slowaxis'])
            self.scantype = np.array(f['scantype'])
            self.poni1 = np.array(f['poni1'])
            self.poni2 = np.array(f['poni2'])
            self.rot1 = np.array(f['rot1'])
            self.rot2 = np.array(f['rot2'])
            self.rot3 = np.array(f['rot3'])
            self.dist = np.array(f['dist'])
            self.wvl = np.array(f['wvl'])
            self.detpixsz1 = np.array(f['detpixsz1'])
            self.detpixsz2 = np.array(f['detpixsz2'])
            self.detshape = np.array(f['detshape'])
            
        print('Number of translations (detector elements for X-ray imaging): ', self.ntrans)
        print('Number of projections: ', self.nproj)
        print('Fast axis: ', self.fastaxis)
        print('Slow axis: ', self.slowaxis)
        print('Data acquisition strategy: ', self.scantype)
        print('Sample-to-detector distance: ', self.dist)
        print('X-ray wavelength: ', self.wvl)
        print('Detector pixel size: ', self.detpixsz1, self.detpixsz2)
        self.detshape = (self.detshape[0], self.detshape[1])
        print('Detector shape: ', self.detshape)
        print('pyFAI detector calibration parameters: poni1 = %f, poni2 = %f, rot1 = %f, rot2 = %f, rot3 = %f' 
              %(self.poni1, self.poni2, self.rot1, self.rot2, self.rot3))
    
    def setscanprms(self, trans, nproj, scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation'):
        
        '''
        Scan type: Zigzig, Zigzag, Interlaced, ContRot
        '''
        
        self.ntrans = trans; self.nproj = nproj; 
        self.scantype = scantype; self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    
    def readponi(self, poni):
        
        '''
        Read a poni file
        '''
        
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
        
        self.detpixsz1 = detpixsz1; self.detpixsz2 = detpixsz2; self.detshape = shape;
        self.detector = pyFAI.detectors.Detector(pixel1=detpixsz1, pixel2=detpixsz2, max_shape=shape)
    
    def azimint(self):
        
        self.ai = AzimuthalIntegrator(dist=self.dist, poni1=self.poni1, poni2=self.poni2, rot1=self.rot1, 
                                 rot2=self.rot2, rot3=self.rot3, detector=self.detector, wavelength=self.wvl)
    
    def conv1Dto2D(self, I, msk=None, units="q_A^-1"):
        
        
        img_theo = self.ai.calcfrom1d(self.xaxis, I, mask=msk, dim1_unit=units,
                                 correctSolidAngle=False,
                                 polarization_factor=0.95)
        
        return(img_theo)

    def saveprms(self):
        
        fn = '%s\\%s_scan_prms.h5' %(self.savedir, self.dname)

        with h5py.File(fn, 'w') as f:

            f.create_dataset('ntrans', data=self.ntrans)
            f.create_dataset('nproj', data=self.nproj)
            f.create_dataset('fastaxis', data=self.fastaxis)
            f.create_dataset('slowaxis', data=self.slowaxis)
            f.create_dataset('scantype', data=self.scantype)
            f.create_dataset('poni1', data=self.poni1)
            f.create_dataset('poni2', data=self.poni2)
            f.create_dataset('rot1', data=self.rot1)
            f.create_dataset('rot2', data=self.rot2)
            f.create_dataset('rot3', data=self.rot3)
            f.create_dataset('dist', data=self.dist)
            f.create_dataset('wvl', data=self.wvl)
            f.create_dataset('detpixsz1', data=self.detpixsz1)
            f.create_dataset('detpixsz2', data=self.detpixsz2)
            f.create_dataset('detshape', data=self.detshape)
            
            f.close()
        

    def create_nDTomo_phantom(self, npix = 100, nproj = 110):
    
        dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
        spectra = [dpAl, dpCu, dpFe, dpPt, dpZn]
        
        self.xaxis = q
        
        iml = nDphantom_2D(npix, nim = 'Multiple')
        
        imAl, imCu, imFe, imPt, imZn = iml
        
        chemct = nDphantom_3D(npix, use_spectra = 'Yes', spectra = spectra, imgs = iml, indices = 'All',  norm = 'No')    
    
        self.data = np.zeros((chemct.shape[0], nproj, chemct.shape[2]))
        
        for ii in tqdm(range(chemct.shape[2])):
                        
            self.data[:,:,ii] = astra_create_sino(chemct[:,:,ii], theta=np.deg2rad(np.arange(0, 180, 180/nproj))).transpose()
            
        self.ntrans = self.data.shape[0]; self.nproj = self.data.shape[1]; 

