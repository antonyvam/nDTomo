# -*- coding: utf-8 -*-
"""
Class used to integrate XRD-CT using CPU or GPU

@author: Antony
"""

from numpy import sin, deg2rad, pi, concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
from PyQt5.QtCore import pyqtSignal, QThread
import os, time, h5py, fabio, pyFAI

class XRDCT_Squeeze(QThread):
    
    '''
    XRD-CT data integration with CPU or GPU    
    '''
    
    squeeze = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,omega,trans,dio,etime):
        QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; #self.jsonname = jsonname
        self.omega = omega; self.trans = trans
        self.dio = dio; self.etime = etime
        dpath = os.path.join( self.xrdctpath, self.dataset)
        print dpath
        try:
            os.chdir(dpath)		
            self.mask = fabio.open(self.maskname)
            self.mask = array(self.mask.data)
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"
            
    def run(self):

        try:
            self.data = zeros((self.nt*self.na,self.npt_rad-10));
            
            ai = pyFAI.load(self.poniname)        

            print self.xrdctpath, self.dataset, self.prefix
            for ii in range(0,self.nt*self.na):
                    start=time.time()

                    if self.datatype == 'cbf':
                        s = '%s/%s_%.4d.cbf' % (self.dataset,self.prefix,ii)
                    else:
                        s = '%s/%s_%.4d.edf' % (self.dataset,self.prefix,ii)
                    pat = os.path.join(self.xrdctpath, s)
                    if os.path.exists(pat) and ii>0:
                        f = fabio.open(pat)
                        d = array(f.data)
    
                        if self.filt == "No":
                            if self.procunit == "CPU":
                                r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="cython",correctSolidAngle=False, polarization_factor=0.95)
                            elif self.procunit == "GPU":
                                r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="ocl_CSR_gpu",correctSolidAngle=False, polarization_factor=0.95)
                        elif self.filt == "Median":
                            if self.procunit == "CPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="splitpixel", correctSolidAngle=False, polarization_factor=0.95);
                            elif self.procunit == "GPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                        elif self.filt == "trimmed_mean":
                            if self.procunit == "CPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(round(float(self.prc)/2),100-round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95); #"splitpixel"
                            elif self.procunit == "GPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(round(float(self.prc)/2),100-round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95); 
                        elif self.filt == "sigma":
                            if self.procunit == "CPU":
                                r, I, stdf = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95);
                            elif self.procunit == "GPU":
                                r, I, stdf = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);

                        self.data[ii,:] = I[0:self.npt_rad-10]
                        
                        v = round((100.*(ii+1))/(int(self.na)*int(self.nt)))
                        self.progress.emit(v)
                        ii += 1;
                        print s, time.time()-start
                    
            print "Integration done, now saving the data"
            self.r = r[0:self.npt_rad-10]
            self.tth = self.r
            self.tth2q()

            self.writehdf5()
            
            print "Saving data done"   
            self.squeeze.emit()
            
        except:
            print "Something is wrong with the integration..."

    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*sin(deg2rad(0.5*self.tth)))
        self.q = pi*2/self.d
                
    def writehdf5(self):


        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                

#        fn = "%s/%s_squeezed_%s.hdf5" % (self.savepath, self.dataset, self.filt)
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