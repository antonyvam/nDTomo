# -*- coding: utf-8 -*-
"""
Class used to integrate XRD-CT using CPU or GPU

@author: A. Vamvakeros
"""

from numpy import sin, deg2rad, pi, concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
from PyQt5.QtCore import pyqtSignal, QThread
import os, time, h5py, fabio, pyFAI

class XRDCT_Squeeze(QThread):
    
    """
    
    XRD-CT data integration with CPU or GPU   
    
        :prefix: prefix used for the filenames of the experimental data
    
	:dataset: foldername where the experimental data are stored
    
	:xrdctpath: full path where the experimental data are stored
    
	:maskname: detector mask file ('mask.edf')
    
	:poniname: .poni file ('filename.poni')
    
	:na: number of angular steps during the tomographic scan
    
	:nt: number of translation steps during the tomographic scan
    
	:npt_rad: number of bins used for the diffraction pattern (e.g. 2048)
    
	:procunit: processing unit, options are: 'CPU', 'GPU' and 'MultiGPU'
    
	:units:  x axis units for the integrated diffraction patterns, options are: 'q_A^-1' or '2th_deg'
    
	:prc: percentage to be used for the trimmed mean filter
    
	:thres: number of standard deviation values to be used for the adaptive standard deviation filter
    
	:datatype: type of image files, options are: 'edf' or 'cbf'
    
	:savepath: full path where results will be saved
    
	:scantype: scantype, options are: 'Zigzag', 'ContRot' and 'Interlaced'
    
	:energy: X-ray energy in keV		
    
	:jsonname: full path for the .azimint.json file
    
	:omega:	rotation axis motor positions during the tomographic scan (1d array)
    
	:trans:	translation axis motor positions during the tomographic scan (1d array)
    
	:dio: diode values per point during the tomographic scan (1d array)
    
	:etime:	values for exposure time per point during the tomographic scan (1d array)
        
    """
    
    squeeze = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,omega,trans,dio,etime,rebin):
        QThread.__init__(self)
		
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; #self.jsonname = jsonname
        self.omega = omega; self.trans = trans
        self.dio = dio; self.etime = etime; self.rebin = int(rebin)
        dpath = os.path.join( self.xrdctpath, self.dataset)
        print(dpath)
        try:
            os.chdir(dpath)		
            self.mask = fabio.open(self.maskname)
            self.mask = array(self.mask.data)
            
        except:
            print("Cannot open mask file or the xrd-ct dataset directory is wrong")
            
    def run(self):

        """
        
		Initiate the XRD-CT data integration process
        
		"""
        
        try:
            self.data = zeros((self.nt*self.na,self.npt_rad-10));
            
            ai = pyFAI.load(self.poniname)        

            print(self.xrdctpath, self.dataset, self.prefix)
            
            if self.datatype == 'h5':
                pat = '%s%s/%s_0000.h5' %(self.xrdctpath, self.dataset, self.dataset)
                fl = h5py.File(pat, 'r')
                print(pat)
                
            if self.filt == "No":
                if self.procunit == "CPU":
                    Imethod = pyFAI.method_registry.IntegrationMethod(dim = 1, split = "no", algo = "histogram", impl = "cython")
                elif self.procunit == "GPU":
                    Imethod = pyFAI.method_registry.IntegrationMethod(dim = 1, split = "no", algo = "histogram", impl = "opencl")
            else:
                if self.procunit == "CPU":
                    Imethod = pyFAI.method_registry.IntegrationMethod(dim = 2, split = "no", algo = "histogram", impl = "cython")
                elif self.procunit == "GPU":
                    Imethod = pyFAI.method_registry.IntegrationMethod(dim = 2, split = "no", algo = "histogram", impl = "opencl")            
            
            ntot = (self.nt*self.na)/self.rebin
            
            if self.datatype == 'cbf':
                pat = '%s%s/%s_%.4d.cbf' % (self.xrdctpath, self.dataset,self.prefix,0)
            elif self.datatype == 'edf':
                pat = '%s%s/%s_%.4d.edf' % (self.xrdctpath, self.dataset,self.prefix,0)
                
            if os.path.exists(pat):
                
                if self.datatype == 'h5':
                    self.imsize = fl['/entry_0000/measurement/Pilatus/data/'][0]
                else:
                    f = fabio.open(pat)
                    self.imsize = array(f.data)      
                self.imd = zeros((self.imsize.shape[0],self.imsize.shape[1]))
                print(self.imd.shape)
            
            mm = 0
            for ii in range(0,ntot,self.rebin):
                    start=time.time()

                    if self.rebin>1:
                        self.imd = zeros((self.imsize.shape[0],self.imsize.shape[1]))
                        for kk in range(ii,ii+self.rebin):
    
                            if self.datatype == 'cbf':
                                pat = '%s%s/%s_%.4d.cbf' % (self.xrdctpath, self.dataset,self.prefix,kk)
                            elif self.datatype == 'edf':
                                pat = '%s%s/%s_%.4d.edf' % (self.xrdctpath, self.dataset,self.prefix,kk)
                                
                            if os.path.exists(pat):
                                
                                if self.datatype == 'h5':
                                    d = fl['/entry_0000/measurement/Pilatus/data/'][kk]
                                else:
                                    f = fabio.open(pat)
                                    d = array(f.data)
                            self.imd = self.imd + d
                    else:
                        if self.datatype == 'cbf':
                            pat = '%s%s/%s_%.4d.cbf' % (self.xrdctpath, self.dataset,self.prefix,ii)
                        elif self.datatype == 'edf':
                            pat = '%s%s/%s_%.4d.edf' % (self.xrdctpath, self.dataset,self.prefix,ii)
                            
                        if os.path.exists(pat) and ii>0:
                            
                            if self.datatype == 'h5':
                                d = fl['/entry_0000/measurement/Pilatus/data/'][kk]
                            else:
                                f = fabio.open(pat)
                                d = array(f.data)         
                            self.imd = d
    
                    if self.filt == "No":
                        r, I = ai.integrate1d(data=self.imd, npt=self.npt_rad, mask=self.mask, unit=self.units, method=Imethod,correctSolidAngle=False, polarization_factor=0.95)
                    elif self.filt == "Median":
                        r, I = ai.medfilt1d(data=self.imd, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method=Imethod, correctSolidAngle=False, polarization_factor=0.95);
                    elif self.filt == "trimmed_mean":
                        r, I = ai.medfilt1d(data=self.imd, npt_rad=int(self.npt_rad), percentile=(round(float(self.prc)/2),100-round(float(self.prc)/2)), unit=self.units, mask=self.mask, method=Imethod, correctSolidAngle=False, polarization_factor=0.95); #"splitpixel"
                    elif self.filt == "sigma":
                        r, I, stdf = ai.sigma_clip(data=self.imd, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method=Imethod, correctSolidAngle=False, polarization_factor=0.95);

                    self.data[mm,:] = I[0:self.npt_rad-10]
                    mm += 1
                    
                    v = round((100.*(ii+1))/(int(self.na)*int(self.nt)))
                    self.progress.emit(v)
                    print('Frame %d out of %d' %(ii, ntot), time.time()-start)
                    
            print("Integration done, now saving the data")
            self.r = r[0:self.npt_rad-10]
            self.tth = self.r
            self.tth2q()

            self.writehdf5()
            
            print("Saving data done")   
            
            self.squeeze.emit()
            
        except:
            print("Something is wrong with the integration...")

    def tth2q(self):

        """
        
		Convert 2theta to d and q spacing
        
		"""  	
        
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*sin(deg2rad(0.5*self.tth)))
        self.q = pi*2/self.d
                
    def writehdf5(self):

        """
        
		Export the integrated diffraction data from the tomographic scan as a single .hdf5 file.
        The integrated data are saved as a 2D array.
        
		"""  
        
        if self.filt == "No":
            
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
            
        elif self.filt == "Median":
            
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
            
        elif self.filt == "trimmed_mean":
            
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
            
        elif self.filt == "sigma":
        
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)        

        h5f = h5py.File(fn, "w")
        
        h5f.create_dataset('data', data=self.data)
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('d', data=self.d)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('y', data=self.trans)
        h5f.create_dataset('energy', data=self.E)
        h5f.create_dataset('scantype', data=self.scantype)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        
        h5f.close()
    	
        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %fn
        
        os.system(perm)

