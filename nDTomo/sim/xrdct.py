# -*- coding: utf-8 -*-

"""
xrdct: Simulation of 2D XRD-CT Data from 1D Diffraction Patterns

This module provides a virtual acquisition framework for simulating 
2D X-ray diffraction computed tomography (XRD-CT) datasets using 
1D diffraction sinogram data. The class `nDVAcq` allows users to:
- Define acquisition strategies and scan parameters.
- Convert 1D diffraction patterns into 2D diffraction frames.
- Simulate detector readout with optional noise models.
- Save the acquired data in multiple formats (`cbf`, `edf`, `tiff`).
- Export and import scan parameters for reproducibility.

This simulation framework is useful for validating and optimizing 
XRD-CT reconstruction algorithms and processing workflows.

@author: Dr A. Vamvakeros
"""

#%

from nDTomo.methods.noise import addpnoise2D

from tqdm import tqdm
import time, fabio, h5py
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import pyFAI.detectors
from pyFAI.calibrant import get_calibrant
import numpy as np

class nDVAcq():
    
    """
    Simulates the acquisition of 2D X-ray diffraction computed tomography (XRD-CT) data
    using 1D diffraction sinogram data.

    This class enables the simulation of various scan strategies, detector configurations,
    and acquisition parameters. It supports data export in multiple formats and includes
    options for adding Poisson noise to the simulated frames.

    Parameters
    ----------
    mask : np.ndarray or None, optional
        A mask for the detector area (default is None).
    file_format : str, optional
        Format for saving the acquired frames. Options: 'cbf', 'edf', 'tiff' (default is 'cbf').
    fastaxis : str, optional
        Axis along which fast scanning occurs. Options: 'Translation' or 'Rotation' (default is 'Translation').
    slowaxis : str, optional
        Axis along which slow scanning occurs. Options: 'Translation' or 'Rotation' (default is 'Rotation').
    units : str, optional
        Units for the diffraction pattern conversion (default is "q_A^-1").
    """
    
    def __init__(self, mask = None, file_format ='cbf', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1"):
                        
        self.mask = mask; self.file_format = file_format; self.units = units;
        self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    def setdata(self, data, xaxis):
        
        """
        Sets the 1D diffraction sinogram data for acquisition.

        Parameters
        ----------
        data : np.ndarray
            A 3D array of shape (ntrans, nproj, nch), representing 
            the sinogram data.
        xaxis : np.ndarray
            The axis for the 1D diffraction patterns.
        """  
        self.data = data; self.xaxis = xaxis
        self.ntrans = self.data.shape[0]; self.nproj = self.data.shape[1]; 
        
        
    def setdirs(self, savedir, dname):
        """
        Sets the directory and filename for saving the acquired data.

        Parameters
        ----------
        savedir : str
            Directory where the acquired frames will be saved.
        dname : str
            Base filename for the output files.
        """        
        self.savedir = savedir; self.dname = dname
        
    def run(self):
        """
        Starts the virtual acquisition process.
        """        
        self.start_scan()
    
    def start_scan(self, addnoise = 'No', ct = 0.1):
        
        """
        Simulates the data acquisition process.

        The function performs the virtual scan according to the defined 
        scan type, fast axis, and slow axis. It iterates over the 
        projections and translations to generate 2D diffraction frames.

        Parameters
        ----------
        addnoise : bool, optional
            If True, applies Poisson noise to the frames (default is False).
        ct : float, optional
            Scaling factor for the noise model (default is 0.1).
        """
        
        print('Starting the virtual scan')
        print('Fast axis is %s, slow axis is %s' %(self.fastaxis, self.slowaxis))
        print('The raw data will be saved as %s' %(self.file_format))
        
        
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
        
        """
        Saves the current diffraction frame in the specified format.

        Parameters
        ----------
        fn : str
            Filename without extension. The extension is determined 
            by the selected file format.
        """

        if self.file_format == 'edf':
            
            self.write_edf('%s.edf' %fn)
    
        elif self.file_format == 'cbf':
            
            self.write_cbf('%s.cbf' %fn)
            
        elif self.file_format == 'tiff' or self.file_format == 'tif':
            
            self.write_tiff('%s.tiff' %fn)            
            
    def write_edf(self, fn):
        """
        Saves the current frame as an EDF file.

        Parameters
        ----------
        fn : str
            Full path to the output file.
        """        
        fabio.edfimage.EdfImage(self.frame).write(fn)
    
    def write_cbf(self, fn):
        """
        Saves the current frame as a CBF file.

        Parameters
        ----------
        fn : str
            Full path to the output file.
        """        
        fabio.cbfimage.CbfImage(self.frame).write(fn)
    
    def write_tiff(self, fn):
        """
        Saves the current frame as a TIFF file.

        Parameters
        ----------
        fn : str
            Full path to the output file.
        """        
        fabio.tifimage.TifImage(self.frame).write(fn)
    
    def write_h5(self):
        """
        Placeholder method for saving frames in HDF5 format.
        """        
        pass
    
    def readscanprms(self, fn=None):
        """
        Reads the scan parameters from an HDF5 file.

        Parameters
        ----------
        fn : str, optional
            Path to the scan parameters file. If None, uses the default 
            path based on `savedir` and `dname`.
        """        
        if fn is None:
            
            fn = '%s\\%s_scan_prms.h5' %(self.savedir, self.dname)
        
        with h5py.File(fn, 'r') as f:
    
            print(f.keys())        
    
            self.ntrans = np.array(f['ntrans'])
            self.nproj = np.array(f['nproj'])
            self.fastaxis = np.array(f['fastaxis'])
            self.slowaxis = np.array(f['slowaxis'])
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
        print('Sample-to-detector distance: ', self.dist)
        print('X-ray wavelength: ', self.wvl)
        print('Detector pixel size: ', self.detpixsz1, self.detpixsz2)
        self.detshape = (self.detshape[0], self.detshape[1])
        print('Detector shape: ', self.detshape)
        print('pyFAI detector calibration parameters: poni1 = %f, poni2 = %f, rot1 = %f, rot2 = %f, rot3 = %f' 
              %(self.poni1, self.poni2, self.rot1, self.rot2, self.rot3))
    
    def setscanprms(self, trans, nproj, fastaxis = 'Translation', slowaxis = 'Rotation'):
        
        """
        Sets the scan parameters for the simulation.

        Parameters
        ----------
        trans : int
            Number of translation steps (detector elements).
        nproj : int
            Number of projections (angular steps).
        fastaxis : str, optional
            Fast scanning axis. Options: 'Translation', 'Rotation'.
        slowaxis : str, optional
            Slow scanning axis. Options: 'Translation', 'Rotation'.
        """
        
        self.ntrans = trans; self.nproj = nproj; 
        self.fastaxis = fastaxis; self.slowaxis = slowaxis
        
    
    def readponi(self, poni):
        
        """
        Reads detector calibration parameters from a PONI file.

        Parameters
        ----------
        poni : str
            Path to the PONI calibration file.
        """
        
        ai = pyFAI.load(poni)
        self.dist = ai.dist; self.poni1 = ai.poni1; self.poni2 = ai.poni2
        self.rot1 = ai.rot1; self.rot2 = ai.rot2; self.rot3 = ai.rot3        
        self.wvl = ai.wavelength
        self.ImageD11geo = ai.getImageD11()
    
    def setwvl(self, wvl):
        """
        Sets the X-ray wavelength.

        Parameters
        ----------
        wvl : float
            X-ray wavelength in meters.
        """        
        self.wvl = wvl
    
    def setxrdprms(self, dist, poni1, poni2, rot1, rot2, rot3):
        """
        Sets the XRD calibration parameters.

        Parameters
        ----------
        dist : float
            Sample-to-detector distance.
        poni1 : float
            Detector calibration parameter (first coordinate).
        poni2 : float
            Detector calibration parameter (second coordinate).
        rot1 : float
            First rotation parameter.
        rot2 : float
            Second rotation parameter.
        rot3 : float
            Third rotation parameter.
        """        
        self.dist = dist; self.poni1 = poni1; self.poni2 = poni2
        self.rot1 = rot1; self.rot2 = rot2; self.rot3 = rot3
        
    def calcbeamcentre(self):
        """
        Computes the beam center coordinates based on detector parameters.
        """        
        self.c1 = self.poni1/self.detector.pixel1
        self.c2 = self.poni2/self.detector.pixel2
    
    def setdetector(self, detpixsz1= 0.000172, detpixsz2= 0.000172, shape= (1679,1475) ):
        """
        Sets the detector properties.

        Parameters
        ----------
        detpixsz1 : float, optional
            Pixel size along the first dimension (default is 172 µm).
        detpixsz2 : float, optional
            Pixel size along the second dimension (default is 172 µm).
        shape : tuple of int, optional
            Detector shape in pixels (default is (1679, 1475)).
        """        
        self.detpixsz1 = detpixsz1; self.detpixsz2 = detpixsz2; self.detshape = shape;
        self.detector = pyFAI.detectors.Detector(pixel1=detpixsz1, pixel2=detpixsz2, max_shape=shape)
    
    def azimint(self):
        """
        Initializes the azimuthal integrator for diffraction pattern conversion.
        """        
        self.ai = AzimuthalIntegrator(dist=self.dist, poni1=self.poni1, poni2=self.poni2, rot1=self.rot1, 
                                 rot2=self.rot2, rot3=self.rot3, detector=self.detector, wavelength=self.wvl)
    
    def conv1Dto2D(self, I, msk=None, units="q_A^-1"):
        """
        Converts a 1D diffraction pattern into a 2D diffraction image.

        Parameters
        ----------
        I : np.ndarray
            1D diffraction intensity data.
        msk : np.ndarray, optional
            Optional mask for the detector.
        units : str, optional
            Units for the diffraction pattern (default is "q_A^-1").

        Returns
        -------
        np.ndarray
            The generated 2D diffraction image.
        """        
        
        img_theo = self.ai.calcfrom1d(self.xaxis, I, mask=msk, dim1_unit=units,
                                 correctSolidAngle=False,
                                 polarization_factor=0.95)
        
        return(img_theo)

    def saveprms(self):
        """
        Saves the scan parameters to an HDF5 file.
        """        
        fn = '%s\\%s_scan_prms.h5' %(self.savedir, self.dname)

        with h5py.File(fn, 'w') as f:

            f.create_dataset('ntrans', data=self.ntrans)
            f.create_dataset('nproj', data=self.nproj)
            f.create_dataset('fastaxis', data=self.fastaxis)
            f.create_dataset('slowaxis', data=self.slowaxis)
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
        
