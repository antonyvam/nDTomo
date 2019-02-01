# -*- coding: utf-8 -*-
"""
Class to read live XRD-CT integrated data collected with the continuous rotation-translation method

@author: A. Vamvakeros
"""

from PyQt5.QtCore import pyqtSignal, QThread
import os, fabio, h5py
from numpy import zeros, array, arange, mod, ceil, deg2rad, sin, pi
from PThread import Periodic

class Fast_XRDCT_LiveRead(QThread): 
    
    '''
    Read integrated data live    
    '''    
    
    updatedata = pyqtSignal()
    exploredata = pyqtSignal()
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
        self.data = zeros((1,1,1))
                
        dpath = os.path.join( self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = array(self.mask.data)
            
#            if self.filt == "No":
#	            self.sinos = np.zeros((self.nt*self.na,2000));
#            else:
#                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
            self.sinos = zeros((self.nt*self.na,self.npt_rad-10));
            print self.sinos.shape
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"

    def run(self):
        self.nextLineNumber = 1
        
        if self.procunit == "MultiGPU":
        #### For GPU machine
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(1, self.checkForFile)
        else:
        #### For OAR
            self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(5, self.checkForFile)        

    def checkForFile(self):
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
            ### I should be sending a signal here, this should be caught by the GUI and update the figures
            self.updatedata.emit()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))

    def hdfLineScanRead(self):
        with h5py.File(self.nextFile,'r') as f:
            if self.procunit == "MultiGPU":
            #### For GPU  
                if self.filt == "No":
#                    data = f['/entry_0000/PyFAI/process_integrate1d/I'][:] 
#                    self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:]
                    data = f['/entry_0000/PyFAI/process_integrate1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:,0:self.npt_rad-10]           
                elif self.filt == "Median":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]             
                elif self.filt == "trimmed_mean":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
                elif self.filt == "sigma":
                    data = f['/entry_0000/PyFAI/process_sigma_clip/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_sigma_clip/2th'][0:self.npt_rad-10]      
            else:
            #### For OAR
                data = f['/I'][:]
                self.tth = f['/twotheta'][:]
            f.close
            self.tth2q()
            
        if self.nextLineNumber == 1: # this is a fix as we dont know the number of channels or the x scale
            self.data = zeros((self.nt, self.na, data.shape[1] ))
            print self.data.shape
        
        v1 = len(arange(0,self.data.shape[0],2))
        v2 = len(arange(1,self.data.shape[0],2))
        ll = v1 + v2;
               
        if self.scantype == "fast":
            
            ind = int(ceil(float(self.nextLineNumber-1)/2))
            
            if mod(self.nextLineNumber-1,2) == 0:
                self.data[ind,:,:] = data
            else:
                self.data[ll-ind,:] = data #data[::-1]
                
        if self.nextLineNumber == 1:
            self.exploredata.emit()

    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*sin(deg2rad(0.5*self.tth)))
        self.q = pi*2/self.d
        
    def setNextFile(self):
        self.nextLineNumber += 1
        
        if self.procunit == "MultiGPU":
            #### For GPU
            if self.nextLineNumber < self.nt+1:
                self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"  
                
        else: 
            #### For OAR
            if self.nextLineNumber < self.nt+1:
                self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"               

    def setTargetFile(self):
        
        if self.procunit == "MultiGPU":
            #### For GPU        
            if self.nextLineNumber < self.nt:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.nt:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

        else:
            #### For OAR
            if self.nextLineNumber < self.nt:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.nt:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
        
    def writesinos(self):
        
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
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('d', data=self.d)        
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('translations', data=self.trans)      
        h5f.create_dataset('scantype', data=self.scantype)     
        h5f.close()
    	
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        
    def removedata(self):

        if self.procunit == "MultiGPU":
            try:
                fn = "%s/%s*.azim.h5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No .azim.h5 files present in the folder'
        
        else:
            try:
                fn = "%s/%s*linescan*.hdf5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No *linescan*.hdf5 files present in the folder' 