# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:56:29 2017

@author: AV
"""

from __future__ import unicode_literals

from matplotlib import use as u
u('Qt5Agg')

from PyQt5 import QtCore, QtWidgets, QtGui

import sys, time #, scipy.io
import os, h5py, json
import numpy as np
import scipy.optimize as sciopt

try:
    from scipy.misc import imresize, imrotate
except:
    print "Cannot import imresize and/or imrotate"
    
from skimage.transform import radon, iradon, resize, rotate
from scipy.interpolate import interp1d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

try:
    import tomopy
except:
    print "Cannot import tomopy"

try:
    import astra
except:
    print "Cannot import astra"

try:
    import fabio
except:
    print "Cannot import fabio"


try:
    from  silx.opencl.backprojection import Backprojection as fbp
except:
    print "Silx is not installed or there is a problem with pyopencl"


try:
    from diffpy.pdfgetx.pdfconfig import createPDFConfig
    from diffpy.pdfgetx.pdfgetter import PDFGetter
except:
    print "Cannot import pdfgetx - Will not be able to create PDFs"
    
try:
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except:
    pass
    
#%
class FitData(QtCore.QThread):
    fitdone = QtCore.pyqtSignal()
    progress_fit = QtCore.pyqtSignal(int)
    
    def __init__(self,data,roi,Area,Areamin,Areamax,Pos,Posmin,Posmax,FWHM,FWHMmin,FWHMmax):
        QtCore.QThread.__init__(self)
        self.fitdata = data    
        self.phase = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.cen = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.wid = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg1 = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg2 = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.xroi = roi
        self.Area = Area; self.Areamin = Areamin; self.Areamax = Areamax; 
        self.Pos = Pos; self.Posmin = Posmin; self.Posmax = Posmax; 
        self.FWHM = FWHM; self.FWHMmin = FWHMmin; self.FWHMmax = FWHMmax;
        
        
        
        im = np.sum(self.fitdata,axis = 2);im = im/np.max(im);im = np.where(im<0.1,0,im);msk = im; msk = np.where(msk>0,1,0)
        for ii in range(0,self.fitdata.shape[2]):
            self.fitdata[:,:,ii] = self.fitdata[:,:,ii]*msk

    def run(self):
        self.batchfit()
        
    def batchfit(self):
                
        mdp = np.mean(np.mean(self.fitdata, axis=0),axis = 0)
        y = mdp[self.xroi]
        x0 = np.array([float(self.Area), float(self.Pos), float(self.FWHM), 0., float(np.min(y))], dtype=float)
#        x0 = np.array([1., np.mean(self.xroi), 1., 0., np.min(y)], dtype=float)
        param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-np.inf,-np.inf],[float(self.Areamax),float(self.Posmax),float(self.FWHMmax),np.inf,np.inf])
#        param_bounds=([-1,np.mean(self.xroi)-5.,0,-np.inf,-np.inf],[1E3,np.mean(self.xroi)+5,1E3,np.inf,np.inf])
          
        for ii in range(0,self.fitdata.shape[0]):
            for jj in range(0,self.fitdata.shape[1]):
                
                dp = self.fitdata[ii,jj,:]
                           
                try:            
                    params, params_covariance = sciopt.curve_fit(self.gmodel2, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    self.phase[ii,jj] = params[0]
                    self.cen[ii,jj] = params[1]
                    self.wid[ii,jj] = params[2]         
                    self.bkg1[ii,jj] = params[3]   
                    self.bkg2[ii,jj] = params[4]              
           
                except: 
                    self.phase[ii,jj] = 0
                    self.cen[ii,jj] = (param_bounds[0][1] + param_bounds[1][1])/2
                    self.wid[ii,jj] = (param_bounds[0][2] + param_bounds[1][2])/2     
                    self.bkg1[ii,jj] = 0
                    self.bkg2[ii,jj] = 0
                
                v = (100.*(ii+1))/(self.fitdata.shape[0])
                self.progress_fit.emit(v)
                
                self.phase = np.where(self.phase<0,0,self.phase)
                self.res = {'Phase':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2}
        
        self.fitdone.emit()

    def gmodel2(self, x, A, m, w, a, b):
        return A * np.exp( - (x-m)**2/w) + a*x + b

class SinoProcessing(QtCore.QThread):
    snprocdone = QtCore.pyqtSignal()
    progress_sino = QtCore.pyqtSignal(int)
    
    def __init__(self,sinos,sinonorm,ofs,crsr,scantype):
        QtCore.QThread.__init__(self)
        self.sinos_proc = sinos
        self.sinonorm = sinonorm
        self.ofs = ofs
        self.crsr = crsr
        self.scantype = scantype
        
        
    def run(self):
        
        if self.sinonorm == 1:
            self.scalesinos()   
        
        if self.ofs>0:
            self.airrem()

        if self.crsr>0:       
            self.sinocentering()
        
    def airrem(self):
        
        for ii in range(0,self.sinos_proc.shape[1]):
            dpair = (np.mean(self.sinos_proc[0:self.ofs,ii,:],axis = 0) + np.mean(self.sinos_proc[self.sinos_proc.shape[0]-self.ofs:self.sinos_proc.shape[0],ii,:],axis = 0))/2
            self.sinos_proc[:,ii,:] = self.sinos_proc[:,ii,:] - dpair
            
    def scalesinos(self):
        
        ss = np.sum(self.sinos_proc,axis = 2)
        scint = np.zeros((self.sinos_proc.shape[1]))
        # Summed scattering intensity per linescan
        for ii in range(0,self.sinos_proc.shape[1]):
            scint[ii] = np.sum(ss[:,ii])
        # Scale factors
        sf = scint/np.max(scint)
        
        # Normalise the sinogram data    
        for jj in range(0,self.sinos_proc.shape[1]):
            self.sinos_proc[:,jj,:] = self.sinos_proc[:,jj,:]/sf[jj] 
        
    def sinocentering(self):
        
        
                
        di = self.sinos_proc.shape
        if len(di)>2:
            s = np.sum(self.sinos_proc, axis = 2)
        
        cr =  np.arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.1)
    #    nomega = s.shape[1] - 1
        
        xold = np.arange(0,s.shape[0])
        
        st = []; ind = [];
        
        
        for kk in range(0,len(cr)):
            
            xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
            sn = np.zeros((len(xnew),s.shape[1]))
            
            
            for ii in range(0,s.shape[1]):
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)


            re = sn[::-1,-1]
    #        re = sn[:,:1]
            st.append((np.std((sn[:,0]-re)))); ind.append(kk)
    #        plt.figure(1);plt.clf();plt.plot(sn[:,0]);plt.plot(re);plt.pause(0.001);
    
    
        m = np.argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = np.zeros((len(xnew),self.sinos_proc.shape[1],self.sinos_proc.shape[2]))  
            for ll in range(0,self.sinos_proc.shape[2]):
                for ii in range(0,self.sinos_proc.shape[1]):
                    col = self.sinos_proc[:,ii,ll]
                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll])    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
                v = (100.*(mm+1))/self.sinos_proc.shape[2]
                mm = mm + 1
                self.progress_sino.emit(v)
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1,:]
                
#                print ll
                
        elif len(di)==2:
            
            sn = np.zeros((len(xnew),s.shape[1]))    
            for ii in range(0,s.shape[1]):
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1]
            
        self.sinos_proc = sn
        self.snprocdone.emit()
        
class ReconstructData(QtCore.QThread):
    recdone = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    
    def __init__(self,sinos):
        QtCore.QThread.__init__(self)
        self.sinos = sinos
        
    def run(self):

        npr = self.sinos.shape[1]
        self.theta = np.arange(0,180-180./(npr-1),180./(npr-1))
        self.sinos = self.sinos[0:self.sinos.shape[0],0:len(self.theta),:]
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
        self.remring()
        
        self.recdone.emit()
        
    def rec_skimage(self):

        self.bp = np.zeros((self.sinos.shape[0],self.sinos.shape[0],self.sinos.shape[2]))
        
        ist = 0
        ifi = self.sinos.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sinos[:,0:len(self.theta),ii], theta=self.theta, output_size=self.sinos.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
        
    def rec_silx(self):

        
        self.bp = np.zeros((self.sinos.shape[0],self.sinos.shape[0],self.sinos.shape[2]))
        t = fbp(sino_shape=(self.sinos.shape[1],self.sinos.shape[0]),devicetype='CPU')
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sinos.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=np.transpose(self.sinos[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            

    def remring(self):
        
        sz = np.floor(self.bp.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = np.where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = np.where(r>0.98*sz/2,0,self.bp[:,:,ii])

class BatchProcessing(QtCore.QThread):
    snprocdone = QtCore.pyqtSignal()
    progress_sino = QtCore.pyqtSignal(int)
    recdone = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    
    def __init__(self,sinos,sinonorm,ofs,crsr,scantype,output,tth,q,d):
        QtCore.QThread.__init__(self)
        self.sinos_proc = sinos
        self.sinonorm = sinonorm
        self.ofs = ofs
        self.crsr = crsr
        self.scantype = scantype
        self.output = output
        self.tth = tth
        self.d = d
        self.q = q
        
        
    def run(self):
        
        if self.sinonorm == 1:
            self.scalesinos()   
        
        if self.ofs>0:
            self.airrem()

        if self.crsr>0:       
            self.sinocentering()
        
        npr = self.sinos_proc.shape[1]
        self.theta = np.arange(0,180-180./(npr-1),180./(npr-1))
        self.sinos_proc = self.sinos_proc[0:self.sinos_proc.shape[0],0:len(self.theta),:]
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
#        self.rec_skimage()
        
        self.remring()
        
        self.recdone.emit()
        
        self.savedata()
        
    def airrem(self):
        
        for ii in range(0,self.sinos_proc.shape[1]):
            dpair = (np.mean(self.sinos_proc[0:self.ofs,ii,:],axis = 0) + np.mean(self.sinos_proc[self.sinos_proc.shape[0]-self.ofs:self.sinos_proc.shape[0],ii,:],axis = 0))/2
            self.sinos_proc[:,ii,:] = self.sinos_proc[:,ii,:] - dpair
            
    def scalesinos(self):
        
        ss = np.sum(self.sinos_proc,axis = 2)
        scint = np.zeros((self.sinos_proc.shape[1]))
        # Summed scattering intensity per linescan
        for ii in range(0,self.sinos_proc.shape[1]):
            scint[ii] = np.sum(ss[:,ii])
        # Scale factors
        sf = scint/np.max(scint)
        
        # Normalise the sinogram data    
        for jj in range(0,self.sinos_proc.shape[1]):
            self.sinos_proc[:,jj,:] = self.sinos_proc[:,jj,:]/sf[jj] 
        
    def sinocentering(self):
                
        di = self.sinos_proc.shape
        if len(di)>2:
            s = np.sum(self.sinos_proc, axis = 2)
        
        cr =  np.arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.1)
    #    nomega = s.shape[1] - 1
        
        xold = np.arange(0,s.shape[0])
        
        st = []; ind = [];
        
        
        for kk in range(0,len(cr)):
            
            xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
            sn = np.zeros((len(xnew),s.shape[1]))
            
            
            for ii in range(0,s.shape[1]):
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)

            re = sn[::-1,-1]
    #        re = sn[:,:1]
            st.append((np.std((sn[:,0]-re)))); ind.append(kk)
    #        plt.figure(1);plt.clf();plt.plot(sn[:,0]);plt.plot(re);plt.pause(0.001);
    
        m = np.argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = np.zeros((len(xnew),self.sinos_proc.shape[1],self.sinos_proc.shape[2]))  
            for ll in range(0,self.sinos_proc.shape[2]):
                for ii in range(0,self.sinos_proc.shape[1]):
                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll])    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
                v = (100.*(mm+1))/self.sinos_proc.shape[2]
                mm = mm + 1
                self.progress_sino.emit(v)
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1,:]
                
#                print ll
                
        elif len(di)==2:
            
            sn = np.zeros((len(xnew),s.shape[1]))    
            for ii in range(0,s.shape[1]):
                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1]
            
        self.sinos_proc = sn
        self.snprocdone.emit()
        
    def rec_skimage(self):

        self.bp = np.zeros((self.sinos_proc.shape[0],self.sinos_proc.shape[0],self.sinos_proc.shape[2]))
        
        ist = 0
        ifi = self.sinos_proc.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sinos_proc[:,0:len(self.theta),ii], theta=self.theta, output_size=self.sinos_proc.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
        
    def rec_silx(self):

        
        self.bp = np.zeros((self.sinos_proc.shape[0],self.sinos_proc.shape[0],self.sinos_proc.shape[2]))
        t = fbp(sino_shape=(self.sinos_proc.shape[1],self.sinos_proc.shape[0]),devicetype='CPU')
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sinos_proc.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=np.transpose(self.sinos_proc[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
    def remring(self):
        
        sz = np.floor(self.bp.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = np.where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = np.where(r>0.98*sz/2,0,self.bp[:,:,ii])
        try:
            self.bp = np.where(self.bp<0,0,self.bp)
        except:
            print "No negative values in the reconstructed data"
        
    def savedata(self):

        if len(self.output)>0:
    
            h5f = h5py.File(self.output, "w")
            h5f.create_dataset('Sinograms', data=self.sinos_proc)
            h5f.create_dataset('twotheta', data=self.tth)
            h5f.create_dataset('q', data=self.q)
            h5f.create_dataset('d', data=self.d)
            
            dims = self.bp.shape
            if len(dims)>1:
                h5f.create_dataset('Reconstructed', data=self.bp)            
#            h5f.create_dataset('scantype', data=self.scantype)
            h5f.close()
        
            print 'Dataset %s has been processed and saved' %(self.output)
            
            perm = 'chmod 777 %s' %self.output
            try:
                os.system(perm)  
            except:
                'Wrote the data but problem with permissions'

                
        
class AlignImages(QtCore.QThread):
    aligndone = QtCore.pyqtSignal()
    progress_al = QtCore.pyqtSignal(int)
    
    def __init__(self,theta,iman,bpx):
        QtCore.QThread.__init__(self)
        self.theta = theta 
        self.iman = iman
        self.bpx = bpx
        
    def run(self):
                   
        #Rotate the micro-CT image
        
        dia = []; ang = []
        kk = 0
        fpa = radon(self.iman, theta=self.theta, circle=True)
        roi_an = np.arange(-11,11,0.1)
        
        for ii in roi_an:
            
            thetan = self.theta + ii
            
            imn = iradon(fpa, theta=thetan, circle=True)
            imn = np.where(imn<0,0,imn)
            imn = imn/np.max(imn)        
            
            
            dia.append(np.mean((np.abs(imn-self.bpx))))
            ang.append(ii)
            kk = kk + 1
            
            v = (100.*(kk+1))/(len(roi_an))
            self.progress_al.emit(v)
            
        m = np.argmin(dia)
        print ang[m]
        
        thetan = self.theta + ang[m]
        self.iman = iradon(fpa, theta=thetan, circle=True)
        self.iman = np.where(self.iman<0,0,self.iman)
        self.iman = self.iman/np.max(self.iman)   
        
        self.aligndone.emit()        
        

class NormaliseABSCT(QtCore.QThread):
    normdone = QtCore.pyqtSignal()
    progress_norm = QtCore.pyqtSignal(int)
    
    def __init__(self,ifn,flat_im,dark_im,roixi,roixf,roiyi,roiyf):
        QtCore.QThread.__init__(self)
        self.ifn = ifn
        self.flat_im = flat_im
        self.dark_im = dark_im        
        
        self.roixi = roixi
        self.roixf = roixf
        self.roiyi = roiyi
        self.roiyf = roiyf
        
    def run(self):


        if len(self.ifn)>0:
            self.i = fabio.open(self.ifn)
            self.nd = self.i.nframes
    
            self.normalise()
            
        
    def normalise(self):

        self.roix = range(self.roixi,self.roixf)
        self.roiy = range(self.roiyi,self.roiyf)
        
        self.roip = range(50,self.nd-50) # The GUI should ask how many extra images we are collecting, in this case it is 100 images
        
        self.r = np.zeros((len(self.roix),len(self.roiy),len(self.roip)))
        
        kk = 0
        for ii in self.roip:
            im = self.i.getframe(ii).data[self.roixi:self.roixf,self.roiyi:self.roiyf]
            self.r[:,:,kk] = np.abs(-np.log((im-self.dark_im[self.roixi:self.roixf,self.roiyi:self.roiyf])/self.flat_im[self.roixi:self.roixf,self.roiyi:self.roiyf]))
            v = (100.*(kk+1))/len(self.roip)
            self.progress_norm.emit(v)
            kk = kk + 1
            
        self.normdone.emit()

class ReconABSCT(QtCore.QThread):
    recabsdone = QtCore.pyqtSignal()
    progressabs = QtCore.pyqtSignal(int)
    progressabsrec = QtCore.pyqtSignal()
    
    def __init__(self,sinos,sc,savepathabs,dataset,absscantype,offset,sino):
        QtCore.QThread.__init__(self)
        self.sinos = np.transpose(sinos,(1,2,0))
#        self.sinos = self.sinos[:,50:self.sinos.shape[1]-50,:] # The GUI should ask how many extra images we are collecting, in this case it is 100 images
        print self.sinos.shape
        self.sc = sc
        print self.sc
        self.savepathabs = savepathabs
        self.dataset = dataset
        self.absscantype = absscantype
        self.offset = offset
        self.sino = sino
        print self.sino.shape
                
        if self.absscantype == "360":
            
            self.sn = np.zeros((self.sino.shape[0],self.sino.shape[1],self.sinos.shape[2]))
            print self.sn.shape
            
            for ii in range(0,self.sinos.shape[2]):
                s1 = self.sinos[0:self.sinos.shape[0]-self.offset,0:self.sinos.shape[1]/2,ii]
                s2 = np.flipud(self.sinos[0:self.sinos.shape[0]-self.offset,self.sinos.shape[1]/2:self.sinos.shape[1],ii])
                self.sn[:,:,ii] = np.concatenate((s1,s2),axis = 0)       
                
                print 'Stitching sinogram %d out of %d' %(ii+1,self.sinos.shape[2])
            self.sinos = self.sn
        
        
        if self.sc<1:
            sn =  imresize(self.sinos[:,:,0], self.sc, 'bilinear')        
            self.sn = np.zeros((sn.shape[0],sn.shape[1],self.sinos.shape[2]))
            
            for ii in range(0,self.sinos.shape[2]):
                self.sn[:,:,ii] = imresize(self.sinos[:,:,ii], self.sc, 'bilinear')
                print 'Resizing image %d out of %d' %(ii,self.sinos.shape[2])
                
        else:
            self.sn = self.sinos
            
        self.sinos = []
        self.crsr = 0.2*self.sn.shape[0]
        self.ofs = 2
        
    def run(self):
        
        self.airrem()
        self.sinocentering()

        npr = self.sn.shape[1]
#        self.theta = np.arange(0,180-180./(npr-1),180./(npr-1))
        
        self.theta = np.arange(0,180,180./(npr))
        print self.theta.shape
#        self.sinos = self.sinos[0:self.sinos.shape[0],0:len(self.theta),:]
        self.total = self.sn.shape[2]
        
        self.bp = np.zeros((self.sn.shape[0],self.sn.shape[0],self.sn.shape[2]))
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
        self.remring()
        self.bp = np.where(self.bp<0,0,self.bp)
        
        self.savedata()
        self.recabsdone.emit()

    def airrem(self):
        
        for ii in range(0,self.sn.shape[1]):
            dpair = (np.mean(self.sn[0:self.ofs,ii,:],axis = 0) + np.mean(self.sn[self.sn.shape[0]-self.ofs:self.sn.shape[0],ii,:],axis = 0))/2
            self.sn[:,ii,:] = self.sn[:,ii,:] - dpair
        print "Air background subtracted"
            
    def sinocentering(self):
        
        di = self.sn.shape
        if len(di)>2:
            s = np.sum(self.sn, axis = 2)
        
        cr =  np.arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.5)
    #    nomega = s.shape[1] - 1
        
        xold = np.arange(0,s.shape[0])
        
        st = []; ind = [];
        
        print "Trying to centre the global sinogram..."
        for kk in range(0,len(cr)):
            
            xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
            sn = np.zeros((len(xnew),s.shape[1]))
            
            
            for ii in range(0,s.shape[1]):
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)


            re = sn[::-1,-1]
    #        re = sn[:,:1]
            st.append((np.std((sn[:,0]-re)))); ind.append(kk)
    #        plt.figure(1);plt.clf();plt.plot(sn[:,0]);plt.plot(re);plt.pause(0.001);
    
            print 'Offset no. %d out of %d' %(kk,len(cr))
    
        m = np.argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = np.zeros((len(xnew),self.sn.shape[1],self.sn.shape[2]))  
            for ll in range(0,self.sn.shape[2]):
                for ii in range(0,self.sn.shape[1]):
                    col = self.sn[:,ii,ll]
                    sn[:,ii,ll] = np.interp(xnew, xold, col)    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
#                v = (100.*(mm+1))/self.sn.shape[2]
                mm = mm + 1
                print 'Centering sinogram %d out of %d' %(ll,self.sn.shape[2])
                
        self.sn = sn
        print "All sinograms centered"
        
    def rec_skimage(self):
        
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sn[:,:,ii], theta=self.theta, output_size=self.sn.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
        
    def rec_silx(self):
          
#        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='CPU')
        
        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='GPU')
        
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=np.transpose(self.sn[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
            
    def remring(self):
        
        sz = np.floor(self.bp.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = np.where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = np.where(r>0.98*sz/2,0,self.bp[:,:,ii])
                
    def remring_single(self):
        
        sz = np.floor(self.bpoi.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        self.bpoi = np.where(r>0.98*sz/2,0,self.bpoi)


    def savedata(self):

        self.output = '%s/%s_rec.hdf5' %(self.savepathabs,self.dataset)
        
        if len(self.output)>0:
    
            h5f = h5py.File(self.output, "w")
            h5f.create_dataset('bpa', data=self.bp)
            
            h5f.close()
        
            print 'Dataset %s has been processed and saved' %(self.output)
            
            perm = 'chmod 777 %s' %self.output
            try:
                os.system(perm)  
            except:
                'Wrote the data but problem with permissions'
                
                
                
                
                
                


class BatchProcessingABS(QtCore.QThread):
    batchrecsignal = QtCore.pyqtSignal()
    progressabs = QtCore.pyqtSignal(int)
    progressabsrec = QtCore.pyqtSignal()
    normdone = QtCore.pyqtSignal()
    progress_norm = QtCore.pyqtSignal(int)
    
    def __init__(self,ifn,flat_im,dark_im,roixi,roixf,roiyi,roiyf,sc,savepathabs,dataset,absscantype,offset,sino):
        QtCore.QThread.__init__(self)
        
        self.ifn = ifn
        self.flat_im = flat_im
        self.dark_im = dark_im        
        
        self.roixi = roixi
        self.roixf = roixf
        self.roiyi = roiyi
        self.roiyf = roiyf
        
        self.sc = sc
        print self.sc
        self.savepathabs = savepathabs
        self.dataset = dataset
        self.absscantype = absscantype
        self.offset = offset
        self.sino = sino
        print self.sino.shape
        
    def run(self):

        if len(self.ifn)>0:
            self.i = fabio.open(self.ifn)
            self.nd = self.i.nframes
    
            self.normalise()
            
        
        self.sinos = np.transpose(self.sinos,(1,2,0))
        print self.sinos.shape

        
        if self.absscantype == "360":
            
            self.sn = np.zeros((self.sino.shape[0],self.sino.shape[1],self.sinos.shape[2]))
            print self.sn.shape
            
            for ii in range(0,self.sinos.shape[2]):
                s1 = self.sinos[0:self.sinos.shape[0]-self.offset,0:self.sinos.shape[1]/2,ii]
                s2 = np.flipud(self.sinos[0:self.sinos.shape[0]-self.offset,self.sinos.shape[1]/2:self.sinos.shape[1],ii])
                self.sn[:,:,ii] = np.concatenate((s1,s2),axis = 0)       
                
                print 'Stitching sinogram %d out of %d' %(ii+1,self.sinos.shape[2])
            self.sinos = self.sn
        
        
        if self.sc<1:
            sn =  imresize(self.sinos[:,:,0], self.sc, 'bilinear')        
            self.sn = np.zeros((sn.shape[0],sn.shape[1],self.sinos.shape[2]))
            
            for ii in range(0,self.sinos.shape[2]):
                self.sn[:,:,ii] = imresize(self.sinos[:,:,ii], self.sc, 'bilinear')
                print 'Resizing image %d out of %d' %(ii,self.sinos.shape[2])
                
        else:
            self.sn = self.sinos
            
        self.sinos = []
        self.crsr = 0.2*self.sn.shape[0]
        self.ofs = 5
        
        self.airrem()
        self.sinocentering()

        npr = self.sn.shape[1]
#        self.theta = np.arange(0,180-180./(npr-1),180./(npr-1))
        
        self.theta = np.arange(0,180,180./(npr))
        print self.theta.shape
#        self.sinos = self.sinos[0:self.sinos.shape[0],0:len(self.theta),:]
        self.total = self.sn.shape[2]
        
        self.bp = np.zeros((self.sn.shape[0],self.sn.shape[0],self.sn.shape[2]))
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
        self.remring()
        self.bp = np.where(self.bp<0,0,self.bp)
        
        self.savedata()
        

    def normalise(self):

        self.roix = range(self.roixi,self.roixf)
        self.roiy = range(self.roiyi,self.roiyf)
        
        self.roip = range(50,self.nd-50) # The GUI should ask how many extra images we are collecting, in this case it is 100 images
        
        self.sinos = np.zeros((len(self.roix),len(self.roiy),len(self.roip)))
        
        kk = 0
        for ii in self.roip:
            im = self.i.getframe(ii).data[self.roixi:self.roixf,self.roiyi:self.roiyf]
            self.sinos[:,:,kk] = np.abs(-np.log((im-self.dark_im[self.roixi:self.roixf,self.roiyi:self.roiyf])/self.flat_im[self.roixi:self.roixf,self.roiyi:self.roiyf]))
            v = (100.*(kk+1))/len(self.roip)
            self.progress_norm.emit(v)
            kk = kk + 1
            
        self.normdone.emit()

    def airrem(self):
        
        for ii in range(0,self.sn.shape[1]):
            dpair = (np.mean(self.sn[0:self.ofs,ii,:],axis = 0) + np.mean(self.sn[self.sn.shape[0]-self.ofs:self.sn.shape[0],ii,:],axis = 0))/2
            self.sn[:,ii,:] = self.sn[:,ii,:] - dpair
        print "Air background subtracted"
            
    def sinocentering(self):
        
        di = self.sn.shape
        if len(di)>2:
            s = np.sum(self.sn, axis = 2)
        
        cr =  np.arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.5)
    #    nomega = s.shape[1] - 1
        
        xold = np.arange(0,s.shape[0])
        
        st = []; ind = [];
        
        print "Trying to centre the global sinogram..."
        for kk in range(0,len(cr)):
            
            xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
            sn = np.zeros((len(xnew),s.shape[1]))
            
            
            for ii in range(0,s.shape[1]):
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)


            re = sn[::-1,-1]
    #        re = sn[:,:1]
            st.append((np.std((sn[:,0]-re)))); ind.append(kk)
    #        plt.figure(1);plt.clf();plt.plot(sn[:,0]);plt.plot(re);plt.pause(0.001);
    
            print 'Offset no. %d out of %d' %(kk,len(cr))
    
        m = np.argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = np.zeros((len(xnew),self.sn.shape[1],self.sn.shape[2]))  
            for ll in range(0,self.sn.shape[2]):
                for ii in range(0,self.sn.shape[1]):
                    col = self.sn[:,ii,ll]
                    sn[:,ii,ll] = np.interp(xnew, xold, col)    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
#                v = (100.*(mm+1))/self.sn.shape[2]
                mm = mm + 1
                print 'Centering sinogram %d out of %d' %(ll,self.sn.shape[2])
                
        self.sn = sn
        print "All sinograms centered"
        
    def rec_skimage(self):
        
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sn[:,:,ii], theta=self.theta, output_size=self.sn.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
        
    def rec_silx(self):
          
#        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='CPU')
        
        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='GPU')
        
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=np.transpose(self.sn[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
            
    def remring(self):
        
        sz = np.floor(self.bp.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = np.where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = np.where(r>0.98*sz/2,0,self.bp[:,:,ii])
                
    def remring_single(self):
        
        sz = np.floor(self.bpoi.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        self.bpoi = np.where(r>0.98*sz/2,0,self.bpoi)


    def savedata(self):

        self.output = '%s/%s_rec.hdf5' %(self.savepathabs,self.dataset)
        
        if len(self.output)>0:
    
            h5f = h5py.File(self.output, "w")
            h5f.create_dataset('bpa', data=self.bp)
            
            h5f.close()
        
            print 'Dataset %s has been processed and saved' %(self.output)
            
            perm = 'chmod 777 %s' %self.output
            try:
                os.system(perm)  
            except:
                'Wrote the data but problem with permissions'
            
            self.batchrecsignal.emit()
                
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.data = np.zeros(())
        self.sinos = np.zeros(())
        self.bp = np.zeros(())
        self.bpa = np.zeros(())
        self.tth = np.zeros(())
        self.xaxis = np.zeros(())
        self.zaxis = np.zeros(())
        self.d = np.zeros(())
        self.q = np.zeros(())
        self.wavel = np.zeros(())
        self.imoi = np.zeros(())
        self.sinonorm = 0
        self.ofs = 0
        self.crsr = 1
        self.scantype = 'zigzag'
        self.hdf_fileName = ''
        self.dproi = 0
        self.c = 3E8
        self.h = 6.620700406E-34        
        self.pathslist = []
        self.pathslistabs = []
        self.pathslistflat = []
        self.pathslistdark = []
        self.ch2 = 1
        self.ch1 = 1
        self.cmap = 'jet'
        self.dark_im = 0; self.flat_im = 0
        self.roixi = 0; self.roixf = 0
        self.roiyi = 0; self.roiyf = 0
        self.projn = 0
        self.offset = 0 
        self.zim = 0
        self.sc = 1
        
        self.Area = 1.; self.Areamin = 0.; self.Areamax = 1000.; 
        self.Pos = 50.; self.Posmin = self.Pos - 5; self.Posmax = self.Pos + 5; 
        self.FWHM = 1.; self.FWHMmin = 0.; self.FWHMmax = 100.;
        
        self.cmap_list = ['viridis','plasma','inferno','magma','cividis','flag', 
            'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Visualiser")

#        self.left = 100
#        self.top = 100
#        self.width = 640
#        self.height = 480
#        self.setGeometry(self.left, self.top, self.width, self.height)

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Open XRD-CT Scan Parametes File', self.loadparfile)             
        self.file_menu.addAction('&Open XRD-CT data', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
#        self.file_menu.addAction('&Open XRD-CT reconstructed images', self.loadxrdct)
        self.file_menu.addAction('&Open micro-CT data', self.loadabsct)           
        self.file_menu.addAction('&Save XRD-CT data', self.savexrdct)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.help_menu.addAction('&About', self.about)        
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        
        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        
        # Add tabs
        self.tabs.addTab(self.tab1,"XRD-CT data")
        self.tabs.addTab(self.tab2,"Absorption correction") 
        self.tabs.addTab(self.tab3,"Peak fitting") 
        self.tabs.addTab(self.tab4,"ABS/PHC-CT data") 
        
        self.tab1.layout = QtWidgets.QGridLayout(self)       
        self.tab2.layout = QtWidgets.QGridLayout(self)
        self.tab3.layout = QtWidgets.QGridLayout(self)
        self.tab4.layout = QtWidgets.QGridLayout(self)

        # set up the mapper
        self.mapperWidget = QtWidgets.QWidget(self)
        self.mapper = MyCanvas()
        self.mapperExplorerDock = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock.setWidget(self.mapperWidget)
        self.mapperExplorerDock.setFloating(False)
        self.mapperToolbar = NavigationToolbar(self.mapper, self)
        
        vbox1 = QtWidgets.QVBoxLayout()
#        vbox1 = QtWidgets.QGridLayout()
        
        vbox1.addWidget(self.mapperToolbar)
        vbox1.addWidget(self.mapper)
        self.mapperWidget.setLayout(vbox1)
       
        #set up the plotter
        self.plotterWidget = QtWidgets.QWidget(self)
        self.plotter = MyCanvas()
        self.plotterExplorerDock = QtWidgets.QDockWidget("Histogram", self)
        self.plotterExplorerDock.setWidget(self.plotterWidget)
        self.plotterExplorerDock.setFloating(False)
        self.plotterToolbar = NavigationToolbar(self.plotter, self)
        
        vbox2 = QtWidgets.QVBoxLayout()
#        vbox2 = QtWidgets.QGridLayout()
        
        vbox2.addWidget(self.plotterToolbar)        
        vbox2.addWidget(self.plotter) # starting row, starting column, row span, column span
        self.plotterWidget.setLayout(vbox2)
        
        self.setCentralWidget(self.tabs)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.plotterExplorerDock)
        
        # set up the abs-ct mapper
        self.mapperWidget_absct = QtWidgets.QWidget(self)
        self.mapper_absct = MyCanvas()
        self.mapperExplorerDock_absct = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock_absct.setWidget(self.mapperWidget_absct)
        self.mapperExplorerDock_absct.setFloating(False)
        self.mapperToolbar_absct = NavigationToolbar(self.mapper_absct, self)
        vbox3 = QtWidgets.QVBoxLayout()        
        vbox3.addWidget(self.mapperToolbar_absct)
        vbox3.addWidget(self.mapper_absct)
        self.mapperWidget_absct.setLayout(vbox3)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock_absct)
        self.mapperExplorerDock_absct.setVisible(False)

        # set up the xrd-ct image mapper
        self.mapperWidget_xrdct = QtWidgets.QWidget(self)
        self.mapper_xrdct = MyCanvas()
        self.mapperExplorerDock_xrdct = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock_xrdct.setWidget(self.mapperWidget_xrdct)
        self.mapperExplorerDock_xrdct.setFloating(False)
        self.mapperToolbar_xrdct = NavigationToolbar(self.mapper_xrdct, self)
        vbox3 = QtWidgets.QVBoxLayout()        
        vbox3.addWidget(self.mapperToolbar_xrdct)
        vbox3.addWidget(self.mapper_xrdct)
        self.mapperWidget_xrdct.setLayout(vbox3)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock_xrdct)
        self.mapperExplorerDock_xrdct.setVisible(False)

                
        ############### TAB 1 -  XRD-CT panel ############### 
        
#        self.ChooseImage = QtWidgets.QLabel(self)
#        self.ChooseImage.setText('XRD-CT data')
#        self.tab1.layout.addWidget(self.ChooseImage,0,0)
#        
#
#        self.ViewDataVol=QtWidgets.QCheckBox("View XRD-CT data",self) # QtWidgets.QPushButton
#        self.ViewDataVol.stateChanged.connect(self.viewdata) #mybutton1.pressed.connect(self.viewdata)     
#        self.tab1.layout.addWidget(self.ViewDataVol,1,0)
        
#        self.ZigZag=QtWidgets.QCheckBox("Zig zag scan",self)
#        self.ZigZag.stateChanged.connect(self.flipcolumns)
#        self.l.addWidget(self.ZigZag,1,1)
                
#        self.ReconstructVol=QtWidgets.QCheckBox("Reconstruct Data Volume",self)
#        self.ReconstructVol.stateChanged.connect(self.fbprec_vol)
#        self.l.addWidget(self.ReconstructVol,2,0)

#        self.label13 = QtWidgets.QLabel(self)
#        self.label13.setText('Perform XRD-CT data integration')
#        self.l.addWidget(self.label13,10,1)

        
#        self.Energy = QtWidgets.QLabel(self)
#        self.Energy.setText('X-ray energy (keV)')
#        self.l.addWidget(self.Energy,1,1)
#                    
##        self.EnergySel = QtWidgets.QSpinBox(self)
##        self.EnergySel.valueChanged.connect(self.selEnergy)
##        self.l.addWidget(self.EnergySel,2,2)        
#
#        self.EnergySel = QtWidgets.QLineEdit(self)
#        self.EnergySel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
#        self.EnergySel.textChanged.connect(self.selEnergy)
#        self.l.addWidget(self.EnergySel,1,2)  

        self.DatasetLabel = QtWidgets.QLabel(self)
        self.DatasetLabel.setText('Dataset:')
        self.tab1.layout.addWidget(self.DatasetLabel,0,0)

        self.DatasetNameLabel = QtWidgets.QLabel(self)
        self.tab1.layout.addWidget(self.DatasetNameLabel,0,1)         

        self.ChooseColormap = QtWidgets.QLabel(self)
        self.ChooseColormap.setText('Choose colormap')
        self.tab1.layout.addWidget(self.ChooseColormap,0,2)
        
        self.ChooseColormapType = QtWidgets.QComboBox(self)
        self.ChooseColormapType.addItems(self.cmap_list)
        self.ChooseColormapType.currentIndexChanged.connect(self.changecolormap)
        self.ChooseColormapType.setMaximumWidth(170)
        self.tab1.layout.addWidget(self.ChooseColormapType,0,3)  
    

        self.XAxisData = QtWidgets.QLabel(self)
        self.XAxisData.setText('Choose x axis')
        self.tab1.layout.addWidget(self.XAxisData,1,0)
        
        self.ChooseXAxis = QtWidgets.QComboBox(self)
        self.ChooseXAxis.addItems(["2theta", "Q", "d"])
        self.ChooseXAxis.currentIndexChanged.connect(self.changeXAxis)
        self.ChooseXAxis.setEnabled(False)
        self.ChooseXAxis.setMaximumWidth(170)
        self.tab1.layout.addWidget(self.ChooseXAxis,1,1)  
        
#        self.LabelExport = QtWidgets.QLabel(self)
#        self.LabelExport.setText('Export local diffraction pattern')
#        self.tab1.layout.addWidget(self.LabelExport,1,2)
        
        self.ExportDPbutton = QtWidgets.QPushButton("Export local diffraction pattern",self)
        self.ExportDPbutton.clicked.connect(self.exportdp)
#        self.ExportDPbutton.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ExportDPbutton,1,2)

        self.ExportIMbutton = QtWidgets.QPushButton("Export image of interest",self)
        self.ExportIMbutton.clicked.connect(self.exportim)
#        self.ExportDPbutton.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ExportIMbutton,1,3)
        

        self.NormaliseSinos=QtWidgets.QCheckBox("Normalise sinograms",self) # QtWidgets.QPushButton
        self.NormaliseSinos.stateChanged.connect(self.norm) #mybutton1.pressed.connect(self.viewdata)     
        self.tab1.layout.addWidget(self.NormaliseSinos,2,0)
        
        self.Air = QtWidgets.QLabel(self)
        self.Air.setText('Air scattering (voxels)')
        self.Air.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.Air,2,1)        
        
#        self.AirSel = QtWidgets.QLineEdit(self)
#        self.AirSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
#        self.AirSel.textChanged.connect(self.selEnergy)
#        self.l.addWidget(self.AirSel,2,2)          
        
        self.airspinbox = QtWidgets.QSpinBox(self)
        self.airspinbox.valueChanged.connect(self.changeAir)
        self.airspinbox.setMinimum(0)
        self.airspinbox.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.airspinbox,2,2)        
        
        self.Centering = QtWidgets.QLabel(self)
        self.Centering.setText('Sinogram centering max offset (voxels)')
        self.tab1.layout.addWidget(self.Centering,2,3)        
                    
        self.crspinbox = QtWidgets.QSpinBox(self)
        self.crspinbox.valueChanged.connect(self.changeCR)
        self.crspinbox.setMinimum(1)
        self.crspinbox.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.crspinbox,2,4)        
        
        
        self.ProcessVol = QtWidgets.QPushButton("Process sinogram data",self)
        self.ProcessVol.clicked.connect(self.sinoproc)
        self.ProcessVol.setEnabled(False)
        self.ProcessVol.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ProcessVol,3,0)
        
        self.progressbar_s = QtWidgets.QProgressBar(self)
        self.progressbar_s.setMaximumWidth(200)
        self.tab1.layout.addWidget(self.progressbar_s,3,1)
                

        self.ReconstructVol = QtWidgets.QPushButton("Reconstruct XRD-CT data",self)
        self.ReconstructVol.clicked.connect(self.fbprec_vol)
        self.ReconstructVol.setEnabled(False)
        self.ReconstructVol.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ReconstructVol,4,0)
        
        self.progressbar = QtWidgets.QProgressBar(self)
        self.progressbar.setMaximumWidth(200)
        self.tab1.layout.addWidget(self.progressbar,4,1)
        
        self.DispData = QtWidgets.QLabel(self)
        self.DispData.setText('Display data')
        self.tab1.layout.addWidget(self.DispData,4,2)
        
        self.ChooseData = QtWidgets.QComboBox(self)
        self.ChooseData.addItems(["Sinogram data", "Reconstructed data"])
        self.ChooseData.currentIndexChanged.connect(self.changedata)
        self.ChooseData.setEnabled(False)
        self.ChooseData.setMaximumWidth(170)
        self.tab1.layout.addWidget(self.ChooseData,4,3)   



        self.label17 = QtWidgets.QLabel(self)
        self.label17.setText('Perform batch XRD-CT reconstruction')
        self.tab1.layout.addWidget(self.label17,5,0)

        self.pbutton13 = QtWidgets.QPushButton("Add XRD-CT dataset",self)
        self.pbutton13.clicked.connect(self.selXRDCTdata)
        self.tab1.layout.addWidget(self.pbutton13,6,0)

        self.pbutton14 = QtWidgets.QPushButton("Remove XRD-CT dataset",self)
        self.pbutton14.clicked.connect(self.remXRDCTdata)
        self.tab1.layout.addWidget(self.pbutton14,7,0)
        
        self.datalist = QtWidgets.QListWidget(self)
        self.tab1.layout.addWidget(self.datalist,6,1,7,1)

        self.pbutton15 = QtWidgets.QPushButton("Perform batch reconstruction",self)
        self.pbutton15.clicked.connect(self.batchXRDCTdata)
        self.tab1.layout.addWidget(self.pbutton15,6,2)

        self.pbutton16 = QtWidgets.QPushButton("Stop reconstruction",self)
        self.pbutton16.clicked.connect(self.batchstopprocessxrdct)
        self.pbutton16.setEnabled(True)
        self.tab1.layout.addWidget(self.pbutton16,6,3)
        
#        # Separate the two panels
#        
#        self.ChooseImage = QtWidgets.QLabel(self)
#        self.ChooseImage.setText('                    ')
#        self.layout.addWidget(self.ChooseImage,0,4)
#        
#        self.ChooseImage = QtWidgets.QLabel(self)
#        self.ChooseImage.setText('                    ')
#        self.layout.addWidget(self.ChooseImage,0,5)
        
        ############### TAB 2 -  micro-CT panel ###############
        
#        self.ChooseImage = QtWidgets.QLabel(self)
#        self.ChooseImage.setText('Micro-CT data')
#        self.tab2.layout.addWidget(self.ChooseImage,0,1)
#        
        
        self.ViewDataVol2=QtWidgets.QCheckBox("View micro-CT images",self) # QtWidgets.QPushButton
        self.ViewDataVol2.stateChanged.connect(self.viewMicroCT) #mybutton1.pressed.connect(self.viewdata)     
        self.tab2.layout.addWidget(self.ViewDataVol2,1,0)
        
        self.Labelz = QtWidgets.QLabel(self)
        self.Labelz.setText('micro-CT image no.')
        self.tab2.layout.addWidget(self.Labelz,2,0)
        
        self.spinbox = QtWidgets.QSpinBox(self)
        self.spinbox.valueChanged.connect(self.changeImage)
        self.spinbox.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.spinbox,2,1)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
#        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.valueChanged[int].connect(self.changeImage)
        self.slider.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.slider,2,2)
        
        self.Labelx = QtWidgets.QLabel(self)
        self.Labelx.setText('XRD-CT image (channel)')
        self.tab2.layout.addWidget(self.Labelx,3,0)
        
        self.spinbox_xrd = QtWidgets.QSpinBox(self)
        self.spinbox_xrd.valueChanged.connect(self.changeImage_xrd)
        self.spinbox_xrd.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.spinbox_xrd,3,1)
        
        self.slider_xrd = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
#        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider_xrd.valueChanged[int].connect(self.changeImage_xrd)
        self.slider_xrd.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.slider_xrd,3,2)
        
        self.Label1 = QtWidgets.QLabel(self)
        self.Label1.setText('micro-CT image for absorption correction')
        self.Label1.setMaximumHeight(20)
        self.tab2.layout.addWidget(self.Label1,4,0)


        self.Label2 = QtWidgets.QLabel(self)
        self.Label2.setText('micro-CT data voxel size')
        self.tab2.layout.addWidget(self.Label2,4,1)

        self.Label3 = QtWidgets.QLabel(self)
        self.Label3.setText('XRD-CT data voxel size')
        self.tab2.layout.addWidget(self.Label3,4,2)
        
        
        self.abscorim = QtWidgets.QSpinBox(self)
        self.abscorim.valueChanged.connect(self.setabscorim)
        self.abscorim.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.abscorim,5,0)
        
        self.abspixsize = QtWidgets.QLineEdit(self)
        self.abspixsize.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.abspixsize.textChanged.connect(self.setabspixsize)
        self.abspixsize.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.abspixsize,5,1)
        
        self.xrdpixsize = QtWidgets.QLineEdit(self)
        self.xrdpixsize.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.xrdpixsize.textChanged.connect(self.setxrdpixsize)
        self.xrdpixsize.setMaximumWidth(150)
        self.tab2.layout.addWidget(self.xrdpixsize,5,2)

     
        self.CheckBox3=QtWidgets.QCheckBox("Align micro-CT and XRD-CT datasets",self) # QtWidgets.QPushButton
        self.CheckBox3.stateChanged.connect(self.AlignData) #mybutton1.pressed.connect(self.viewdata)     
        self.tab2.layout.addWidget(self.CheckBox3,6,0)
     
        self.progressbar_align = QtWidgets.QProgressBar(self)
        self.progressbar_align.setMaximumWidth(200)
        self.tab2.layout.addWidget(self.progressbar_align,6,1)
        
        self.CheckBox4=QtWidgets.QCheckBox("Perform zero order absorption correction",self) # QtWidgets.QPushButton
        self.CheckBox4.stateChanged.connect(self.abscor) #mybutton1.pressed.connect(self.viewdata)     
        self.CheckBox4.setEnabled(False)
        self.tab2.layout.addWidget(self.CheckBox4,7,0)
        

        
        ############### TAB 3 -  Peak fitting ###############
        
        self.Labelbkg = QtWidgets.QLabel(self)
        self.Labelbkg.setText('Background subtraction for a region of interest')
        self.tab3.layout.addWidget(self.Labelbkg,0,0)
        
        self.Channel1 = QtWidgets.QLabel(self)
        self.Channel1.setText('Initial channel')
        self.tab3.layout.addWidget(self.Channel1,1,0)        
                    
        self.crspinbox1 = QtWidgets.QSpinBox(self)
        self.crspinbox1.valueChanged.connect(self.channel_1)
        self.crspinbox1.setMinimum(1)
        self.tab3.layout.addWidget(self.crspinbox1,1,1)              
        
        self.Channel2 = QtWidgets.QLabel(self)
        self.Channel2.setText('Final channel')
        self.tab3.layout.addWidget(self.Channel2,1,2)        
                    
        self.crspinbox2 = QtWidgets.QSpinBox(self)
        self.crspinbox2.valueChanged.connect(self.channel_2)
        self.crspinbox2.setMinimum(1)
        self.tab3.layout.addWidget(self.crspinbox2,1,3)           
        
        self.pbutton1 = QtWidgets.QPushButton("Plot image",self)
        self.pbutton1.clicked.connect(self.createimage)
        self.tab3.layout.addWidget(self.pbutton1,1,4)
        
        self.expimroi = QtWidgets.QPushButton("Export image",self)
        self.expimroi.clicked.connect(self.exportimroi)
        self.tab3.layout.addWidget(self.expimroi,1,5)        
        
        self.LabelFit = QtWidgets.QLabel(self)
        self.LabelFit.setText('Single peak batch fitting')
        self.tab3.layout.addWidget(self.LabelFit,2,0)

        self.Channel3 = QtWidgets.QLabel(self)
        self.Channel3.setText('Initial channel')
        self.tab3.layout.addWidget(self.Channel3,3,0)        
                    
        self.crspinbox3 = QtWidgets.QSpinBox(self)
        self.crspinbox3.valueChanged.connect(self.channel_initial)
        self.crspinbox3.setMinimum(1)
        self.tab3.layout.addWidget(self.crspinbox3,3,1)              
        
        self.Channel4 = QtWidgets.QLabel(self)
        self.Channel4.setText('Final channel')
        self.tab3.layout.addWidget(self.Channel4,3,2)        
                    
        self.crspinbox4 = QtWidgets.QSpinBox(self)
        self.crspinbox4.valueChanged.connect(self.channel_final)
        self.crspinbox4.setMinimum(1)
        self.tab3.layout.addWidget(self.crspinbox4,3,3)   

        self.pbutton2 = QtWidgets.QPushButton("Autofill constraints",self)
        self.pbutton2.clicked.connect(self.autofill)
        self.tab3.layout.addWidget(self.pbutton2,3,4)
        
        self.LabelTypePeak = QtWidgets.QLabel(self)
        self.LabelTypePeak.setText('Function')
        self.tab3.layout.addWidget(self.LabelTypePeak,4,0)
        
        self.ChooseFunction = QtWidgets.QComboBox(self)
        self.ChooseFunction.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt"])
        self.ChooseFunction.currentIndexChanged.connect(self.profile_function)
        self.ChooseFunction.setEnabled(True)
        self.tab3.layout.addWidget(self.ChooseFunction,4,1)   

        self.LabelTypeOpt = QtWidgets.QLabel(self)
        self.LabelTypeOpt.setText('Optimization')
        self.tab3.layout.addWidget(self.LabelTypeOpt,4,2)
        
        self.ChooseOpt = QtWidgets.QComboBox(self)
        self.ChooseOpt.addItems(["Constained", "Unconstrained"])
        self.ChooseOpt.currentIndexChanged.connect(self.profile_function)
        self.ChooseOpt.setEnabled(True)
        self.tab3.layout.addWidget(self.ChooseOpt,4,3)   
        
        self.LabelLiveOpt = QtWidgets.QLabel(self)
        self.LabelLiveOpt.setText('Live Plot')
        self.tab3.layout.addWidget(self.LabelLiveOpt,4,4)
        
        self.ChooseLive = QtWidgets.QComboBox(self)
        self.ChooseLive.addItems(["No", "Yes"])
#        self.ChooseOpt.currentIndexChanged.connect(self.profile_function)
        self.ChooseLive.setEnabled(True)
        self.tab3.layout.addWidget(self.ChooseLive,4,5) 
        
        self.LabelAreaPeak = QtWidgets.QLabel(self)
        self.LabelAreaPeak.setText('Area')
        self.tab3.layout.addWidget(self.LabelAreaPeak,5,0)

        self.AreaSel = QtWidgets.QLineEdit(self)
        self.AreaSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.AreaSel.setText(str(self.Area))
        self.AreaSel.textChanged.connect(self.selArea)
        self.tab3.layout.addWidget(self.AreaSel,5,1)  
        
        self.LabelAreaMinPeak = QtWidgets.QLabel(self)
        self.LabelAreaMinPeak.setText('Min')
        self.tab3.layout.addWidget(self.LabelAreaMinPeak,5,2)

        self.AreaMinSel = QtWidgets.QLineEdit(self)
        self.AreaMinSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.AreaMinSel.setText(str(self.Areamin));
        self.AreaMinSel.textChanged.connect(self.selAreaMin)
        self.tab3.layout.addWidget(self.AreaMinSel,5,3)          
        
        self.LabelAreaMaxPeak = QtWidgets.QLabel(self)
        self.LabelAreaMaxPeak.setText('Max')
        self.tab3.layout.addWidget(self.LabelAreaMaxPeak,5,4)

        self.AreaMaxSel = QtWidgets.QLineEdit(self)
        self.AreaMaxSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.AreaMaxSel.setText(str(self.Areamax));
        self.AreaMaxSel.textChanged.connect(self.selAreaMax)
        self.tab3.layout.addWidget(self.AreaMaxSel,5,5)          
        
        
        self.LabelPositionPeak = QtWidgets.QLabel(self)
        self.LabelPositionPeak.setText('Position')
        self.tab3.layout.addWidget(self.LabelPositionPeak,6,0)

        self.PosSel = QtWidgets.QLineEdit(self)
        self.PosSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.PosSel.setText(str(self.Pos));
        self.PosSel.textChanged.connect(self.selPos)
        self.tab3.layout.addWidget(self.PosSel,6,1)  


        self.LabelPositionMinPeak = QtWidgets.QLabel(self)
        self.LabelPositionMinPeak.setText('Min')
        self.tab3.layout.addWidget(self.LabelPositionMinPeak,6,2)

        self.PositionMinSel = QtWidgets.QLineEdit(self)
        self.PositionMinSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.PositionMinSel.setText(str(self.Posmin));
        self.PositionMinSel.textChanged.connect(self.selPosMin)
        self.tab3.layout.addWidget(self.PositionMinSel,6,3)          
        
        self.LabelPositionMaxPeak = QtWidgets.QLabel(self)
        self.LabelPositionMaxPeak.setText('Max')
        self.tab3.layout.addWidget(self.LabelPositionMaxPeak,6,4)

        self.PositionMaxSel = QtWidgets.QLineEdit(self)
        self.PositionMaxSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.PositionMaxSel.setText(str(self.Posmax));        
        self.PositionMaxSel.textChanged.connect(self.selPosMax)
        self.tab3.layout.addWidget(self.PositionMaxSel,6,5)  

        self.LabelFWHMPeak = QtWidgets.QLabel(self)
        self.LabelFWHMPeak.setText('FWHM')
        self.tab3.layout.addWidget(self.LabelFWHMPeak,7,0)

        self.FWHMSel = QtWidgets.QLineEdit(self)
        self.FWHMSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.FWHMSel.setText(str(self.FWHM));           
        self.FWHMSel.textChanged.connect(self.selFWHM)
        self.tab3.layout.addWidget(self.FWHMSel,7,1)  
        
        self.LabelFWHMMinPeak = QtWidgets.QLabel(self)
        self.LabelFWHMMinPeak.setText('Min')
        self.tab3.layout.addWidget(self.LabelFWHMMinPeak,7,2)

        self.FWHMMinSel = QtWidgets.QLineEdit(self)
        self.FWHMMinSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.FWHMMinSel.setText(str(self.FWHMmin));                   
        self.FWHMMinSel.textChanged.connect(self.selFWHMMin)
        self.tab3.layout.addWidget(self.FWHMMinSel,7,3)          
        
        self.LabelFWHMMaxPeak = QtWidgets.QLabel(self)
        self.LabelFWHMMaxPeak.setText('Max')
        self.tab3.layout.addWidget(self.LabelFWHMMaxPeak,7,4)

        self.FWHMMaxSel = QtWidgets.QLineEdit(self)
        self.FWHMMaxSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.FWHMMaxSel.setText(str(self.FWHMmax));              
        self.FWHMMaxSel.textChanged.connect(self.selFWHMMax)
        self.tab3.layout.addWidget(self.FWHMMaxSel,7,5)  

        self.pbutton_fit = QtWidgets.QPushButton("Perform batch peak fitting",self)
        self.pbutton_fit.clicked.connect(self.batchpeakfit)
        self.tab3.layout.addWidget(self.pbutton_fit,8,0)
        
        self.progressbar_fit = QtWidgets.QProgressBar(self)
        self.tab3.layout.addWidget(self.progressbar_fit,8,1)

        self.pbutton_stop = QtWidgets.QPushButton("Stop",self)
        self.pbutton_stop.clicked.connect(self.stopfit)
        self.tab3.layout.addWidget(self.pbutton_stop,8,3)
        
        self.LabelRes = QtWidgets.QLabel(self)
        self.LabelRes.setText('Display peak fitting results')
        self.tab3.layout.addWidget(self.LabelRes,9,0)
        
        self.ChooseRes = QtWidgets.QComboBox(self)
        self.ChooseRes.addItems(['Phase','Position', 'FWHM', 'Background_polynomial_const1', 'Background_polynomial_const2'])
        self.ChooseRes.currentIndexChanged.connect(self.plot_fit_results)
        self.ChooseRes.setEnabled(False)
        self.tab3.layout.addWidget(self.ChooseRes,9,1)   
        

        self.LabelCMin = QtWidgets.QLabel(self)
        self.LabelCMin.setText('Min')
        self.tab3.layout.addWidget(self.LabelCMin,9,2)
        
        self.CMinSel = QtWidgets.QLineEdit(self)
        self.CMinSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.CMinSel.textChanged.connect(self.selCMin)
        self.tab3.layout.addWidget(self.CMinSel,9,3)  

        self.LabelCMax = QtWidgets.QLabel(self)
        self.LabelCMax.setText('Max')
        self.tab3.layout.addWidget(self.LabelCMax,9,4)

        self.CMaxSel = QtWidgets.QLineEdit(self)
        self.CMaxSel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.CMaxSel.textChanged.connect(self.selCMax)
        self.tab3.layout.addWidget(self.CMaxSel,9,5)  


        self.pbutton_expfit = QtWidgets.QPushButton("Export fit results",self)
        self.pbutton_expfit.clicked.connect(self.savefitresults)
        self.pbutton_expfit.setEnabled(False)
        self.tab3.layout.addWidget(self.pbutton_expfit,10,0)

        ############### TAB 4 -  Absoption/phase constrast CT ###############

        self.DatasetLabelA = QtWidgets.QLabel(self)
        self.DatasetLabelA.setText('Dataset:')
        self.tab4.layout.addWidget(self.DatasetLabelA,0,0)

        self.DatasetNameLabelA = QtWidgets.QLabel(self)
        self.tab4.layout.addWidget(self.DatasetNameLabelA,0,1)   
        

        self.pbutton_abs = QtWidgets.QPushButton("Tomographic data path",self)
        self.pbutton_abs.clicked.connect(self.read_absdata)
        self.tab4.layout.addWidget(self.pbutton_abs,1,0)
        
        self.pbutton_abspath = QtWidgets.QLineEdit(self)
#        self.pbutton_abspath.textChanged.connect(self.calibrantpath2)
        self.pbutton_abspath.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.pbutton_abspath,1,1)


        self.pbutton_flat = QtWidgets.QPushButton("Flat field data path",self)
        self.pbutton_flat.clicked.connect(self.read_flat)
        self.tab4.layout.addWidget(self.pbutton_flat,2,0)
        
        self.pbutton_flatpath = QtWidgets.QLineEdit(self)
#        self.pbutton_abspath.textChanged.connect(self.calibrantpath2)
        self.pbutton_flatpath.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.pbutton_flatpath,2,1)

        self.pbutton_dark = QtWidgets.QPushButton("Dark current data path",self)
        self.pbutton_dark.clicked.connect(self.read_dark)
        self.tab4.layout.addWidget(self.pbutton_dark,3,0)
        
        self.pbutton_darkpath = QtWidgets.QLineEdit(self)
#        self.pbutton_abspath.textChanged.connect(self.calibrantpath2)
        self.pbutton_darkpath.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.pbutton_darkpath,3,1)

        self.pbuttonsave = QtWidgets.QPushButton("Data save directory",self)
        self.pbuttonsave.clicked.connect(self.selSavepath)
        self.tab4.layout.addWidget(self.pbuttonsave,4,0)
        
        self.savedatapath = QtWidgets.QLineEdit(self)
        self.savedatapath.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.savedatapath,4,1)
        

        self.Labelroixstart = QtWidgets.QLabel(self)
        self.Labelroixstart.setText('Row start')
        self.tab4.layout.addWidget(self.Labelroixstart,5,0)        
                    
        self.roixspinbox1 = QtWidgets.QSpinBox(self)
        self.roixspinbox1.valueChanged.connect(self.select_rowi)
        self.roixspinbox1.setMinimum(0)
        self.roixspinbox1.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.roixspinbox1,5,1)    

        self.Labelroixend = QtWidgets.QLabel(self)
        self.Labelroixend.setText('Row end')
        self.tab4.layout.addWidget(self.Labelroixend,5,2)        
                    
        self.roixspinbox2 = QtWidgets.QSpinBox(self)
        self.roixspinbox2.valueChanged.connect(self.select_rowf)
        self.roixspinbox2.setMinimum(0)
        self.roixspinbox2.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.roixspinbox2,5,3)    
        

        self.Labelroiystart = QtWidgets.QLabel(self)
        self.Labelroiystart.setText('Column start')
        self.tab4.layout.addWidget(self.Labelroiystart,6,0)        
                    
        self.roiyspinbox1 = QtWidgets.QSpinBox(self)
        self.roiyspinbox1.valueChanged.connect(self.select_coli)
        self.roiyspinbox1.setMinimum(0)
        self.roiyspinbox1.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.roiyspinbox1,6,1)    

        self.Labelroiyend = QtWidgets.QLabel(self)
        self.Labelroiyend.setText('Column end')
        self.tab4.layout.addWidget(self.Labelroiyend,6,2)        
                    
        self.roiyspinbox2 = QtWidgets.QSpinBox(self)
        self.roiyspinbox2.valueChanged.connect(self.select_colf)
        self.roiyspinbox2.setMinimum(0)
        self.roiyspinbox2.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.roiyspinbox2,6,3)    


        self.Labelproj = QtWidgets.QLabel(self)
        self.Labelproj.setText('Projection')
        self.tab4.layout.addWidget(self.Labelproj,7,0)        
                    
        self.projspinbox = QtWidgets.QSpinBox(self)
        self.projspinbox.valueChanged.connect(self.select_proj)
        self.projspinbox.setMinimum(0)
        self.projspinbox.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.projspinbox,7,1)   

        self.PlotRadio = QtWidgets.QPushButton("Plot radiograph",self)
        self.PlotRadio.clicked.connect(self.plot_radiograph)
        self.PlotRadio.setEnabled(False)
#        self.NormAbsVol.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.PlotRadio,7,2)
       
        self.DatasetLabelA2 = QtWidgets.QLabel(self)
        self.DatasetLabelA2.setText('Type of scan')
        self.tab4.layout.addWidget(self.DatasetLabelA2,8,0)
        
        self.ChooseScantype = QtWidgets.QComboBox(self)
        self.ChooseScantype.addItems(["0 - 180 deg", "0 - 360 deg"])
        self.ChooseScantype.currentIndexChanged.connect(self.changescantype)
        self.ChooseScantype.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.ChooseScantype,8,1)
        
        self.Labelofs = QtWidgets.QLabel(self)
        self.Labelofs.setText('Offset (pixels)')
        self.tab4.layout.addWidget(self.Labelofs,8,2)        
                    
        self.ofsspinbox = QtWidgets.QSpinBox(self)
        self.ofsspinbox.valueChanged.connect(self.select_ofs)
        self.ofsspinbox.setMinimum(0)
        self.ofsspinbox.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.ofsspinbox,8,3)
        
        self.Labelzim = QtWidgets.QLabel(self)
        self.Labelzim.setText('Image (z)')
        self.tab4.layout.addWidget(self.Labelzim,9,0)        
                    
        self.zimspinbox = QtWidgets.QSpinBox(self)
        self.zimspinbox.valueChanged.connect(self.select_zim)
        self.zimspinbox.setMinimum(0)
        self.zimspinbox.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.zimspinbox,9,1)   
        

        self.labelst = QtWidgets.QPushButton("Plot sinogram",self)
        self.labelst.clicked.connect(self.plot_stitched)
        self.labelst.setEnabled(False)
#        self.NormAbsVol.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.labelst,9,2)


        self.NormAbsVol = QtWidgets.QPushButton("Normalise tomographic data volume",self)
        self.NormAbsVol.clicked.connect(self.norm_adata)
        self.NormAbsVol.setEnabled(False)
#        self.NormAbsVol.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.NormAbsVol,10,0)
        
        self.progressabsbarn = QtWidgets.QProgressBar(self)
        self.progressabsbarn.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.progressabsbarn,10,1)
        
        
        self.Rescale = QtWidgets.QLabel(self)
        self.Rescale.setText('Rescale sinograms (scale factor):')
        self.tab4.layout.addWidget(self.Rescale,11,0)        
        
        self.rescalesinos = QtWidgets.QLineEdit(self)
        self.rescalesinos.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.rescalesinos.textChanged.connect(self.setscale)
        self.rescalesinos.setMaximumWidth(150)
        self.tab4.layout.addWidget(self.rescalesinos,11,1)
        

        self.ReconstructAbsVol = QtWidgets.QPushButton("Reconstruct tomographic data volume",self)
        self.ReconstructAbsVol.clicked.connect(self.reconabsvol)
        self.ReconstructAbsVol.setEnabled(False)
        self.tab4.layout.addWidget(self.ReconstructAbsVol,12,0)
        
        self.progressabsbar = QtWidgets.QProgressBar(self)
        self.progressabsbar.setMaximumWidth(400)
        self.tab4.layout.addWidget(self.progressabsbar,12,1)

        self.pbutton_stop_abs = QtWidgets.QPushButton("Stop",self)
        self.pbutton_stop_abs.clicked.connect(self.stopabsrec)
        self.pbutton_stop_abs.setEnabled(False)
        self.tab4.layout.addWidget(self.pbutton_stop_abs,12,2)
        

        self.Labelz = QtWidgets.QLabel(self)
        self.Labelz.setText('CT image no.')
        self.tab4.layout.addWidget(self.Labelz,13,0)
        
        self.spinboxabs = QtWidgets.QSpinBox(self)
        self.spinboxabs.valueChanged.connect(self.changeImageN)
        self.spinboxabs.setMaximumWidth(150)
        self.spinboxabs.setEnabled(False)
        self.tab4.layout.addWidget(self.spinboxabs,13,1)
        
        self.sliderabs = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
#        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sliderabs.valueChanged[int].connect(self.changeImageN)
        self.sliderabs.setMaximumWidth(150)
        self.sliderabs.setEnabled(False)
        self.tab4.layout.addWidget(self.sliderabs,13,2)


#        self.labelbatch = QtWidgets.QLabel(self)
#        self.labelbatch.setText('Perform batch ABS/PHC-CT reconstruction')
#        self.tab4.layout.addWidget(self.labelbatch,14,0)
#
#        self.pbuttonbatchabs = QtWidgets.QPushButton("Add ABS/PHC-CT dataset",self)
#        self.pbuttonbatchabs.clicked.connect(self.selABSCTdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchabs,15,0)
#
#        self.pbuttonbatchabs2 = QtWidgets.QPushButton("Remove ABS/PHC-CT dataset",self)
#        self.pbuttonbatchabs2.clicked.connect(self.remABSCTdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchabs2,16,0)
#        
#        self.datalistabs = QtWidgets.QListWidget(self)
##        self.tab4.layout.addWidget(self.datalistabs,15,1)
#        self.tab4.layout.addWidget(self.datalistabs,15,1,16,1)
##        self.datalistabs.setMaximumHeight(20)
##
##        self.pbuttonbatchflat = QtWidgets.QPushButton("Add flat field dataset",self)
##        self.pbuttonbatchflat.clicked.connect(self.selFlatdata)
##        self.tab4.layout.addWidget(self.pbuttonbatchflat,15,2)
##
##        self.pbuttonbatchflat2 = QtWidgets.QPushButton("Remove flat field dataset",self)
##        self.pbuttonbatchflat2.clicked.connect(self.remFlatdata)
##        self.tab4.layout.addWidget(self.pbuttonbatchflat2,16,2)
##        
##        self.datalistflat = QtWidgets.QListWidget(self)
##        self.tab4.layout.addWidget(self.datalistflat,15,3)
##        self.datalistflat.setMaximumHeight(20)
##        
##
##        self.pbuttonbatchdark = QtWidgets.QPushButton("Add dark current dataset",self)
##        self.pbuttonbatchdark.clicked.connect(self.selDarkdata)
##        self.tab4.layout.addWidget(self.pbuttonbatchdark,15,4)
##
##        self.pbuttonbatchdark2 = QtWidgets.QPushButton("Remove dark current dataset",self)
##        self.pbuttonbatchdark2.clicked.connect(self.remDarkdata)
##        self.tab4.layout.addWidget(self.pbuttonbatchdark2,16,4)
##        
##        self.datalistdark = QtWidgets.QListWidget(self)
##        self.tab4.layout.addWidget(self.datalistdark,15,5)
##        self.datalistdark.setMaximumHeight(20)
#        
#
#
#        self.pbuttonbatchflat = QtWidgets.QPushButton("Add flat field dataset",self)
#        self.pbuttonbatchflat.clicked.connect(self.selFlatdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchflat,15,2)
#
#        self.pbuttonbatchflat2 = QtWidgets.QPushButton("Remove flat field dataset",self)
#        self.pbuttonbatchflat2.clicked.connect(self.remFlatdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchflat2,16,2)
#        
#        self.datalistflat = QtWidgets.QListWidget(self)
#        self.tab4.layout.addWidget(self.datalistflat,15,3,16,1)
#        
#
#        self.pbuttonbatchdark = QtWidgets.QPushButton("Add dark current dataset",self)
#        self.pbuttonbatchdark.clicked.connect(self.selDarkdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchdark,15,4)
#
#        self.pbuttonbatchdark2 = QtWidgets.QPushButton("Remove dark current dataset",self)
#        self.pbuttonbatchdark2.clicked.connect(self.remDarkdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchdark2,16,4)
#        
#        self.datalistdark = QtWidgets.QListWidget(self)
#        self.tab4.layout.addWidget(self.datalistdark,15,5,16,1)
#        
#
#        self.pbuttonbatchabs = QtWidgets.QPushButton("Perform batch processing",self)
#        self.pbuttonbatchabs.clicked.connect(self.batchABSCTdata)
#        self.tab4.layout.addWidget(self.pbuttonbatchabs,15,6)
#
#        self.pbuttonbatchabsstop = QtWidgets.QPushButton("Stop reconstruction",self)
#        self.pbuttonbatchabsstop.clicked.connect(self.batchstopprocessxrdct)
#        self.pbuttonbatchabsstop.setEnabled(True)
#        self.tab4.layout.addWidget(self.pbuttonbatchabsstop,16,6)
#        




#####

#        self.main_widget.setFocus()
#        self.setCentralWidget(self.main_widget)

        self.tabs.setFocus()
        self.setCentralWidget(self.tabs)
                
        self.tab1.setLayout(self.tab1.layout)     
        self.tab2.setLayout(self.tab2.layout)     
        self.tab3.setLayout(self.tab3.layout)     
        self.tab4.setLayout(self.tab4.layout)     
        
#        self.setLayout(layout)
        self.show()
        
        self.xaxislabel = '2theta'

        self.selectedVoxels = np.empty(0,dtype=object)
        self.selectedChannels = np.empty(0,dtype=object)
    
    
    ####################### XRD-CT #######################

    def exportdp(self):
        
        if len(self.hdf_fileName)>0 and len(self.dproi)>0:
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_%s_%s.hdf5" %(s,str(self.row),str(self.col))
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.dproi)
            if len(self.tth)>0:
                h5f.create_dataset('twotheta', data=self.tth)
            if len(self.q)>0:
                h5f.create_dataset('q', data=self.q)
            if len(self.d)>0:
                h5f.create_dataset('d', data=self.d)
            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            try:
                xy = np.column_stack((self.tth,self.dproi*1E3))
                sn = "%s_%s_%s_twotheta.asc" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
            
                xy = np.column_stack((self.q,self.dproi*1E3))
                sn = "%s_%s_%s_q.asc" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
                
                xy = np.column_stack((self.d,self.dproi*1E3))
                sn = "%s_%s_%s_d.asc" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
                
                xy = np.column_stack((self.tth,self.dproi*1E3))
                sn = "%s_%s_%s_twotheta.xy" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
                
                xy = np.column_stack((self.q,self.dproi*1E3))
                sn = "%s_%s_%s_q.xy" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
                
                xy = np.column_stack((self.d,self.dproi*1E3))
                sn = "%s_%s_%s_d.xy" %(s,str(self.row),str(self.col))
                np.savetxt(sn,xy)
                perm = 'chmod 777 %s' %sn
                os.system(perm) 
                
            except:
                print "Something is wrong with the local diffraction pattern"
                
            
        else:
            print "Something is wrong with the data"
        
    def exportim(self):
        
        if len(self.hdf_fileName)>0 and len(self.imoi)>0:
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_channel_%s.hdf5" %(s,str(self.nx))
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.imoi)
            h5f.create_dataset('Channel', data=self.nx)            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            sn = "%s_channel_%s.png" %(s,str(self.nx))
            plt.imsave(sn,self.imoi)
                        
        else:
            print "Something is wrong with the data"
            
    def selEnergy(self, value):
        self.E = value

    def changeAir(self, value):
        self.ofs = value
        
    def changeCR(self, value):
        self.crsr = value
        
    def viewdata(self,s):
        if s == 2 and self.data.shape[2]>0:
            print self.data.shape
            self.explore()

    def norm(self,s):
        if s == 2:
            self.sinonorm = 1
        else:
            self.sinonorm = 0
            
    def tth2q(self):
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1000)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
    
    def changeXAxis(self,ind):
#        self.tth2q()
        if ind == 0:
            self.xaxis = self.tth
            self.xaxislabel = '2theta'
        elif ind == 1:
            self.xaxis = self.q
            self.xaxislabel = 'Q'
        elif ind == 2:
            self.xaxis = self.d
            self.xaxislabel = 'd'
        self.update()
        
    def changecolormap(self,ind):
        
        self.cmap = self.cmap_list[ind]
        print self.cmap
        try:
            self.update()
        except: 
            pass
            
    def explore(self):
                
        self.map_data = self.mapper.axes.imshow(np.mean(self.data,axis=2),cmap=self.cmap)
        title = 'Mean Sinogram'
        self.mapper.axes.set_title(title, fontstyle='italic')
        self.mapper.fig.canvas.mpl_connect('button_press_event', self.onMapClick)
        self.mapper.fig.canvas.mpl_connect('motion_notify_event', self.onMapMoveEvent)        
        
        
        self.cb = self.mapper.fig.colorbar(self.map_data)
        
        self.mapper.show()
        self.mapper.draw()  
        
        self.mdp = np.mean(np.mean(self.data,axis=1),axis=0)
        self.sdp = np.sum(np.sum(self.data,axis=1),axis=0)
        self.dproi = self.mdp
        self.histogramCurve = self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="b")
        self.activeCurve = self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="r")
        
        ########
        self.vCurve = self.plotter.axes.axvline(x=0, color="k")
        #######
        
#        self.selectedDataSetList.addItem('mean')
        self.plotter.axes.legend()
        self.plotter.axes.set_title("Mean diffraction pattern", fontstyle='italic')
#        self.activeCurve[0].set_visible(False)
#        self.plot_fig.canvas.mpl_connect('button_press_event',self.onPlotClick)        
#        self.plot_fig.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)
#        self.plot_cid = self.plotter.figure.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)
        
        self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotClick)       
        self.plot_cid = self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)   
        self.plotter.show()        
        self.plotter.draw()        
                
        
    def onMapClick(self, event): # SDMJ version
        if event.button == 1:
            if event.inaxes:
                self.col = int(event.xdata.round())
                self.row = int(event.ydata.round())
                self.dproi = self.data[self.row,self.col,:]
                self.histogramCurve[0].set_data(self.xaxis, self.dproi) 
                self.histogramCurve[0].set_label(str(self.row)+','+str(self.col))
                self.histogramCurve[0].set_visible(True)          
                self.plotter.axes.legend()
                
#                self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
#                self.plotter.axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

#                self.selectedDataSetList.addItem(np.str([row,col]))
#                self.selectedDataSetList.setCurrentIndex(self.selectedDataSetList.count()-1)
#                self.selectedDataSetList.update()
                
#            self.plot_fig.canvas.mpl_disconnect(self.plot_cid) 
#            self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)

            else:
                self.selectedVoxels = np.empty(0,dtype=object)
#                self.selectedDataSetList.clear()
#                self.currentCurve = 0
                self.plotter.axes.clear() # not fast
                self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern')
#                self.selectedDataSetList.addItem('mean')
                self.plotter.axes.legend()                
                self.plotter.draw() 

        if event.button == 3:
            if event.inaxes:
                self.histogramCurve[0].set_visible(False)          
                
                self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
                self.plotter.axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

    def onMapMoveEvent(self, event): # SDMJ version
        if event.inaxes:
            
            col = int(event.xdata.round())
            row = int(event.ydata.round())
            
            dproi = self.data[row,col,:]
            self.activeCurve[0].set_data(self.xaxis, dproi) 
            self.mapper.axes.relim()
            self.activeCurve[0].set_label(str(row)+','+str(col))
#            self.mapper.axes.autoscale(enable=True, axis='both', tight=True)#.axes.autoscale_view(False,True,True)
            self.activeCurve[0].set_visible(True)
            self.plotter.axes.set_ylim(0,np.max(dproi))

        else:
            self.activeCurve[0].set_visible(False)
            self.activeCurve[0].set_label('')
            
#            self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
            self.plotter.axes.set_ylim(0,np.max(self.dproi))
            
        self.plotter.axes.legend()
        self.plotter.draw()
    
    
    def onPlotMoveEvent(self, event):
        if event.inaxes:
            
            x = event.xdata
            
            if self.xaxislabel == 'd':
                self.nx = len(self.xaxis) - np.searchsorted(self.xaxisd, [x])[0]
            else:
                self.nx = np.searchsorted(self.xaxis, [x])[0] -1
            
            if self.nx<0:
                self.nx = 0
            elif self.nx>len(self.xaxis):
                self.nx = len(self.xaxis)-1

            
            self.selectedChannels = self.nx;
            self.mapper.axes.clear() # not fast
            self.imoi = self.data[:,:,self.nx]
#            self.map_data = self.mapper.axes.imshow(self.imoi,cmap='jet')
            self.map_data = self.mapper.axes.imshow(self.imoi,cmap = self.cmap)
            title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            self.mapper.axes.set_title(title)

            ############       
#            self.mapper.fig.delaxes(self.mapper.fig.axes[1])
#            self.cb.remove()
#            self.cb = self.mapper.fig.colorbar(self.map_data)
            ############
            
            
            self.mapper.draw()             
            
            self.vCurve.set_xdata(x) 
                
            self.plotter.draw()

    def onPlotClick(self, event):

        if event.button == 1:
            self.plotter.fig.canvas.mpl_disconnect(self.plot_cid)   

        elif event.button == 3:
            self.plot_cid = self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)
            self.plotter.show()  
            self.plotter.draw()   
            
    def update(self):
        self.mapper.axes.clear() # not fast
        # this bit is messy
        if (not self.selectedChannels):
            self.mapper.axes.imshow(np.mean(self.data,2),cmap=self.cmap)
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.mapper.axes.imshow(self.data[:,:,self.selectedChannels],cmap=self.cmap)
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.mapper.axes.imshow(np.mean(self.data[:,:,self.selectedChannels],2),cmap=self.cmap)
                title = self.name+' '+'mean of selected channels'
        self.mapper.axes.set_title(title)
        self.mapper.show()
        self.mapper.draw()  
            
    def flipcolumns(self):
        self.data[:,1::2,:] = self.data[::-1,1::2,:]    
        self.update()

    def sinoproc(self): #(self,s):    

        self.ProcessVol.setEnabled(False)
        self.ProcSinos = SinoProcessing(self.sinos,self.sinonorm,self.ofs,self.crsr,self.scantype)
        self.ProcSinos.start()            
        self.ProcSinos.progress_sino.connect(self.progressbar_s.setValue)
        self.ProcSinos.snprocdone.connect(self.updatesdata)

    def updatesdata(self):
        
        self.data = self.ProcSinos.sinos_proc
        self.ProcessVol.setEnabled(True)
        self.update()
        
    def fbprec_vol(self): #(self,s):    

        try:
            self.sinos = self.ProcSinos.sinos_proc
        except:
            self.sinos = self.data
        
        self.ReconstructVol.setEnabled(False)
        self.Rec = ReconstructData(self.data) # or self.data
        self.Rec.start()            
        self.Rec.progress.connect(self.progressbar.setValue)
        self.Rec.recdone.connect(self.updatedata)
                
    def updatedata(self):
        self.bp = self.Rec.bp
        try:
            self.bp = np.where(self.bp<0,0,self.bp)
        except:
            print "No negative values in the reconstructed data"
        self.data = self.bp
        self.ChooseData.setCurrentIndex(1)
        self.update()
        self.ReconstructVol.setEnabled(True)
        self.ChooseData.setEnabled(True)     
    
    def changedata(self,ind):
        if ind == 1:
            self.data = self.bp
            print "Should be reconstructed data displayed"
        else:
            self.data = self.sinos
            print "Should be sinogram data displayed"
            
        self.update()



    def selXRDCTdata(self):
        
        self.dialog = FileDialog()
        if self.dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.paths = (self.dialog.selectedFiles())
        
        for ii in range(0,len(self.paths)):
            oldpath = str(self.paths[ii])
            try:
                newpath = oldpath.split("/")
                self.datasetname = newpath[-1]
                
                try: 
                    newfile = self.datasetname.split(".hdf5")
                    newfile = newfile[0]
                    if len(self.datasetname)>0 and len(newfile)>0:
                        self.datalist.addItem(self.datasetname)    
                        
                        newpath = oldpath.split(self.datasetname)
                        newpath = newpath[0]
                        self.pathslist.append(newpath)  
                        
                except:
                    print('Something is wrong with the selected datasets')
            except:
                pass
                
    def remXRDCTdata(self):
        
        for item in self.datalist.selectedItems():
            self.datalist.takeItem(self.datalist.row(item))
#        self.datalist.takeItem()


    def batchXRDCTdata(self):

        # Need to disable some buttons
        
        datasets = [] 
        for index in xrange(self.datalist.count()):
            datasets.append(self.datalist.item(index).text())
            
        self.datasets = datasets
#        print self.datasets[0]
        
        self.BatchProc = []       
        self.progressbarbatch = []
        self.labels = []
        for ii in range(0,len(self.datasets)):
            
            self.dataset = self.datasets[ii]
            self.prefix = self.dataset
            
            os.chdir(self.pathslist[ii])
            self.hdf_fileName = self.datasets[ii]
            self.loadxrdct()
    
            filename = self.prefix.split(".hdf5")
            filename = filename[0]
                    
            output = '%s/%s_processed.hdf5' %(self.pathslist[ii],filename)
            print output
        
            self.BatchProc.append(BatchProcessing(self.data,self.sinonorm,self.ofs,self.crsr,self.scantype,output,self.tth,self.q,self.d))

                        
            self.progressbarbatch.append(QtWidgets.QProgressBar(self))
            self.tab1.layout.addWidget(self.progressbarbatch[ii],6+ii,5)
            
            self.labels.append(QtWidgets.QLabel(self))
            self.labels[ii].setText(self.dataset)
            self.tab1.layout.addWidget(self.labels[ii],6+ii,6)

            self.BatchProc[ii].start()                
            self.BatchProc[ii].progress.connect(self.progressbarbatch[ii].setValue)
            
            
    def batchstopprocessxrdct(self):
        
        for ii in range(0,len(self.datasets)):
                        
            self.BatchProc[ii].terminate()
#            self.progressbarbatch[ii].setValue(0)
            self.tab1.layout.removeWidget(self.progressbarbatch[ii])
            self.progressbarbatch[ii].deleteLater()
            self.progressbarbatch[ii] = None
            
            self.tab1.layout.removeWidget(self.labels[ii])
            self.labels[ii].deleteLater()
            self.labels[ii] = None
                    
            

        
#    def tth2d(self):
#        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
    
    ####################### micro-CT #######################

    def viewMicroCT(self,s):
        if s == 2:
            try:
                self.explore_absct()
            except:
                print 'No micro-CT data'
        else:
            try:
                dim = self.bp.shape
                if len(dim) == 3:
                    self.data = self.bp
                else:
                    self.data = self.sinos
                    
                try:
                    self.histogramCurve.pop(0).remove()
                    self.activeCurve.pop(0).remove()
                    self.vCurve.remove()
                except:
                    	pass
#                self.plotter.axes.clear()

                self.mapperExplorerDock_absct.setVisible(False)
                self.mapperExplorerDock_xrdct.setVisible(False)
                self.mapperExplorerDock.setVisible(True)                        
                self.plotterExplorerDock.setVisible(True)
                self.explore()
            except:
                print 'No XRD-CT data'
            
    def explore_absct(self):
        
        self.data = self.bpa
        
        dim = self.data.shape
        
        if len(dim)==2:
            self.mapper.axes.clear()
            self.mapper.axes.imshow(self.data,cmap='jet')
            title = 'micro-CT image'
            self.mapper.axes.set_title(title, fontstyle='italic')
    
            self.mapper.show()
            self.mapper.draw()          
            
        elif len(dim)==3:
            
            self.mapperExplorerDock.setVisible(False)                        
            self.plotterExplorerDock.setVisible(False)
            self.mapperExplorerDock_absct.setVisible(True)
            self.mapperExplorerDock_xrdct.setVisible(True)

                
            self.mapper_absct.axes.clear()
            self.mapper_absct.axes.imshow(np.mean(self.data,axis=2),cmap='jet')
            title = 'Mean micro-CT image'
            self.mapper_absct.axes.set_title(title, fontstyle='italic')
            self.mapper_absct.show()
            self.mapper_absct.draw()  
            
            self.mapper_xrdct.axes.clear()
            self.mapper_xrdct.axes.imshow(np.mean(self.bp,axis=2),cmap='jet')
            title = 'Mean XRD-CT image'
            self.mapper_xrdct.axes.set_title(title, fontstyle='italic')
            self.mapper_xrdct.show()
            self.mapper_xrdct.draw()              
        
    def changeImage(self, value):
        
        dim = self.data.shape
        if len(dim)==3:
            if value>self.bpa.shape[2]:
                value = self.bpa.shape[2]
            elif value<0:
                value = 0
            self.spinbox.setValue(value)
            self.slider.setValue(value)
            
            self.selectedChannels = value;
            self.mapper_absct.axes.clear() # not fast
            self.mapper_absct.axes.imshow(self.bpa[:,:,value],cmap='jet')
            title = "Micro-CT image no. %d" % (value)
            self.mapper_absct.axes.set_title(title)
            self.mapper_absct.show()
            self.mapper_absct.draw()  

    def changeImage_xrd(self, value):
        
        dim = self.bp.shape
        if len(dim)==3:
            if value>self.bp.shape[2]:
                value = self.bp.shape[2]
            elif value<0:
                value = 0
            self.spinbox_xrd.setValue(value)
            self.spinbox_xrd.setValue(value)
            
            self.selectedChannels = value;
            self.mapper_xrdct.axes.clear() # not fast
            self.mapper_xrdct.axes.imshow(self.bp[:,:,value],cmap='jet')
            title = "XRD-CT image channel %d" % (value)
            self.mapper_xrdct.axes.set_title(title)
            self.mapper_xrdct.show()
            self.mapper_xrdct.draw()  
            
    def setxrdpixsize(self, value):
        self.pxsxrd = np.asarray(value , dtype="float64")    
        
    def setabspixsize(self, value):
        self.pxsabs = np.asarray(value , dtype="float64")    
        
    def setabscorim(self,value):
        
        dim = self.bpa.shape
        if len(dim)==3:
            self.ima = self.bpa[:,:,value]
        elif len(dim)==2:
            self.ima = self.bpa
            
    def AlignData(self,s):
        if s==2:
        
            # Rescale the micro-ct image        
            r = self.pxsabs/self.pxsxrd;
            print r
            self.ima = imresize(self.ima, r, 'bilinear')
            self.bpx = np.sum(self.bp,axis=2)
            if self.ima.shape[0]<self.bpx.shape[0]:
                print 'hello'
                self.bpan = np.zeros((self.bpx.shape[0],self.bpx.shape[1]))
                self.bpan[0:self.ima.shape[0],0:self.ima.shape[1]] = self.ima
                self.ima = self.bpan
            elif self.ima.shape[0]>self.bpx.shape[0]:
#                self.bpxn = np.zeros((self.ima.shape[0],self.ima.shape[1]))
#                self.bpxn[0:self.bpx.shape[0],0:self.bpx.shape[1]] = self.bpx
#                self.bpx = self.bpxn;
                self.bpan = np.zeros((self.bpx.shape[0],self.bpx.shape[1]))
                ofs = np.floor((self.ima.shape[0]-self.bpx.shape[0])/2.)
                self.bpan = self.ima[int(ofs):self.ima.shape[0]-int(ofs)-1,int(ofs):self.ima.shape[1]-int(ofs)-1]
                self.ima = self.bpan
                
                if self.ima.shape[0]<self.bpx.shape[0]:
                    self.bpan = np.zeros((self.bpx.shape[0],self.bpx.shape[1]))
                    self.ima = np.array(self.bpa,dtype=float)
                    self.ima = imresize(self.bpa, r, 'bilinear')
                    ofs = np.floor((self.ima.shape[0]-self.bpx.shape[0])/2.)
                    self.bpan = self.ima[int(ofs):self.ima.shape[0]-int(ofs),int(ofs):self.ima.shape[1]-int(ofs)]
                    self.ima = self.bpan                    
            
            # Normalise the images
            
            self.bpx = self.bpx/np.max(self.bpx)
            m = float(np.max(self.ima))
            self.ima = self.ima/m
            
            im = np.concatenate((self.ima, self.bpx), axis=1)
            plt.figure(6);plt.clf();plt.imshow(im, cmap = 'jet');plt.title('Rescaled micro-CT image and global XRD-CT image');plt.colorbar();plt.pause(0.01);
            # Align the two images
            
            roi_ii = np.arange(-30,31);
            roi_jj = np.arange(-30,31);
            
            kk = 0
            di = []; indc = []; indr = []; 
            for ii in roi_ii:
                for jj in roi_jj:
                    self.iman = np.zeros((self.ima.shape[0],self.ima.shape[1]))
                                
                    if ii<0 and jj<0:
                        imoi = self.ima[0-ii::,0-jj::]
                        self.iman[0:imoi.shape[0],0:imoi.shape[1]] = imoi
                    elif ii<0 and jj>=0:
                        imoi = self.ima[0-ii::,0:self.ima.shape[1]-jj]
                        self.iman[0:imoi.shape[0],0+jj:imoi.shape[1]+jj]  = imoi
                    elif ii>=0 and jj>=0:
                        imoi = self.ima[0:self.ima.shape[0]-ii,0:self.ima.shape[1]-jj]
                        self.iman[0+ii:imoi.shape[0]+ii,0+jj:imoi.shape[1]+jj] = imoi
                    elif ii>=0 and jj<0:
                        imoi = self.ima[0:self.ima.shape[0]-ii,0-jj::]
                        self.iman[0+ii:imoi.shape[0]+ii,0:imoi.shape[1]] = imoi
                    
                    
                    di.append(np.mean((np.abs(self.iman-self.bpx))))
                    indc.append(jj)
                    indr.append(ii)
                    
                    kk = kk + 1;
            
            di = np.array(di)
            m = np.argmin(di)
            print indr[m], indc[m]
            
            ii = indr[m]; jj = indc[m];
            self.iman = np.zeros((self.ima.shape[0],self.ima.shape[1]))
            if ii<0 and jj<0:
                imoi = self.ima[0-ii::,0-jj::]
                self.iman[0:imoi.shape[0],0:imoi.shape[1]] = imoi
            elif ii<0 and jj>=0:
                imoi = self.ima[0-ii::,0:self.ima.shape[1]-jj]
                self.iman[0:imoi.shape[0],0+jj:imoi.shape[1]+jj]  = imoi
            elif ii>=0 and jj>=0:
                imoi = self.ima[0:self.ima.shape[0]-ii,0:self.ima.shape[1]-jj]
                self.iman[0+ii:imoi.shape[0]+ii,0+jj:imoi.shape[1]+jj] = imoi
            elif ii>=0 and jj<0:
                imoi = self.ima[0:self.ima.shape[0]-ii,0-jj::]
                self.iman[0+ii:imoi.shape[0]+ii,0:imoi.shape[1]] = imoi
    
            im = np.concatenate((self.iman/np.max(self.iman), self.bpx), axis=1)
            plt.figure(7);plt.clf();plt.imshow(im, cmap = 'jet');plt.title('Rescaled micro-CT image and global XRD-CT image');plt.colorbar();plt.pause(0.01);
            
            #Rotate the micro-CT image
            npr = self.sinos.shape[1] # this is the number of angles in the xrd-ct data
            theta = np.linspace(0., 180., npr, endpoint=False)
            
            self.Align = AlignImages(theta,self.iman,self.bpx)
            self.Align.start()
            self.Align.progress_al.connect(self.progressbar_align.setValue)
            self.Align.aligndone.connect(self.update_adata)
             

#            dia = []; ang = []
#            kk = 0
##            roi_an = np.arange(-21,21,0.1);
#            roi_an = np.arange(-11,11,0.1);
#            for ii in roi_an:
##                imn = imrotate(self.iman,ii,'bilinear');
##                imn = imn/np.max(imn)
#                
#                thetan = theta + ii
#                fpa = radon(self.iman, theta=theta, circle=True)
#                imn = iradon(fpa, theta=thetan, circle=True)
#                
##                plt.figure(7);plt.clf();plt.imshow(imn, cmap = 'jet');plt.colorbar();plt.pause(0.01);
#                
#                dia.append(np.mean((np.abs(imn-self.bpx))))
#                ang.append(ii)
#                kk = kk + 1
#                
#                v = (100.*(kk+1))/(len(roi_an))
#                self.progressbar_abscor.setValue(v)
#                
#            plt.figure(7);plt.clf();plt.plot(dia);plt.pause(0.01);
#            m = np.argmin(dia)
#            print ang[m]
#            
#            self.iman = imrotate(self.iman,ang[m],'bilinear');
#            self.iman = np.asarray(self.iman, dtype="float64")
#            self.iman = self.iman/np.max(self.iman)
#            im = np.concatenate((self.iman, self.bpx), axis=1)
#            plt.figure(8);plt.clf();plt.imshow(im, cmap = 'jet');plt.title('Aligned rescaled micro-CT image and global XRD-CT image');plt.colorbar();plt.pause(0.01);
    
            
        else:
            self.ima = np.array(self.bpa,dtype=float)

                
    def update_adata(self):
        
        self.iman = self.Align.iman
        im = np.concatenate((self.iman, self.bpx), axis=1)
        plt.figure(8);plt.clf();plt.imshow(im, cmap = 'jet');plt.title('Aligned rescaled micro-CT image and global XRD-CT image');plt.colorbar();plt.pause(0.01);
        self.CheckBox4.setEnabled(True)
        
    def abscor(self,s):
        
        if s==2:
            npr = self.sinos.shape[1] # this is the number of angles in the xrd-ct data
            theta = np.linspace(0., 180., npr, endpoint=False)
            
            fpa = radon(self.iman, theta=theta, circle=True)
            
            fpa = fpa/np.max(fpa)
            sroi = np.sum(self.sinos,axis=2)
            sroi = sroi/np.max(sroi)
            
            fpa = radon(self.iman, theta=theta, circle=True)
            fpa = np.exp(-fpa*0.000001*self.pxsxrd)
            
            scor = np.sum(self.sinos,axis=2)/fpa
            
            npr = scor.shape[1]
            theta = np.linspace(0., 180., npr, endpoint=False)
            print scor.shape,theta.shape
            
            self.bpcor = iradon(scor, theta=theta, circle=True)
            self.bpcor = np.where(self.bpcor<0,0,self.bpcor)
            self.bpcor = self.bpcor/np.max(self.bpcor)
    #        im = np.concatenate((self.iman, self.bpx, self.bpcor), axis=1)
            plt.figure(8);plt.clf();plt.imshow(self.bpcor, cmap = 'jet');plt.title('Global absoprtion corrected XRD-CT image');plt.colorbar();plt.pause(0.01);

    ####################### Peak fitting #######################

    def channel_1(self, value):
        self.ch1 = value

    def channel_2(self, value):
        self.ch2 = value

    def createimage(self):
        
        if self.ch2<self.ch1:
            self.ch2 = self.ch1
            self.crspinbox2.setValue(self.ch2)
            
        self.mapper.axes.clear() # not fast
        roi = np.arange(self.ch1,self.ch2)
        self.imroi = np.sum(self.data[:,:,roi],axis=2)-len(roi)*np.mean(self.data[:,:,[self.ch1,self.ch2]],axis=2)
        self.imroi = np.where(self.imroi<0,0,self.imroi)
        self.map_data = self.mapper.axes.imshow(self.imroi,cmap='jet')
        title = "From %s = %.3f to %s = %.3f " % (self.xaxislabel, self.xaxis[self.ch1], self.xaxislabel, self.xaxis[self.ch2])
        self.mapper.axes.set_title(title)

        try:
            self.cb.remove()
        except:
            pass
        self.cb = self.mapper.fig.colorbar(self.map_data)
        cl = self.cb.get_clim()
        self.CMin = cl[0]; self.CMax = cl[1]
        self.CMinSel.setText(str(self.CMin))  
        self.CMaxSel.setText(str(self.CMax))      
        
        self.mapper.draw()   
        

    def exportimroi(self):
        
        if len(self.hdf_fileName)>0 and len(self.imroi)>0:
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_roi_%sto%s.hdf5" %(s,str(self.ch1),str(self.ch2))
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.imroi)
            h5f.create_dataset('ChannelInitial', data=self.ch1)  
            h5f.create_dataset('ChannelFinal', data=self.ch2)  
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
                        
        else:
            print "Something is wrong with the data"
        
    def profile_function(self,ind):
        if ind == 0:
            self.peaktype = "Gaussian"
            print "Gaussian profile"
        elif ind == 1:
            self.peaktype = "Lorentzian"
            print "Lorentzian profile"
        elif ind == 2:
            self.peaktype = "Pseudo"
            print "Pseudo-Voigt profile"


    def channel_initial(self, value):
        self.pos1 = value

    def channel_final(self, value):
        self.pos2 = value

    def autofill(self):
        self.Pos = (self.pos1 + self.pos2)/2.
        self.Posmin = self.pos1
        self.Posmax = self.pos2
        
        self.PosSel.setText(str(self.Pos));
        self.PositionMinSel.setText(str(self.Posmin))
        self.PositionMaxSel.setText(str(self.Posmax))     
        
        self.roi = np.arange(self.Posmin,self.Posmax+1)
        
        try:
            self.Area = np.ceil(np.sum(self.mdp[self.roi])-(self.pos2 - self.pos1)*(self.mdp[self.Posmin]+self.mdp[self.Posmax])/2.)
            self.AreaSel.setText(str(self.Area))
            
            self.Areamax = 10*self.Area
            self.AreaMaxSel.setText(str(self.Areamax))
            
            self.FWHM = (self.Posmax - self.Posmin)
            self.FWHMSel.setText(str(self.FWHM))
            
            self.FWHMmax = self.FWHM*10
            self.FWHMMaxSel.setText(str(self.FWHMmax))
            
        except:
            print 'No data'
        
    def selArea(self,value):
        self.Area = float(value)
        print self.Area

    def selAreaMin(self,value):
        self.Areamin = float(value)
        print self.Areamin

    def selAreaMax(self,value):
        self.Areamax = float(value)
        print self.Areamax

    def selPos(self,value):
        self.Pos = float(value)
        print self.Pos
    
    def selPosMin(self,value):
        self.Posmin = float(value)
        print self.Posmin

    def selPosMax(self,value):
        self.Posmax = float(value)
        print self.Posmax

    def selFWHM(self,value):
        self.FWHM = float(value)
        print self.FWHM

    def selFWHMMin(self,value):
        self.FWHMmin = float(value)
        print self.FWHMmin

    def selFWHMMax(self,value):
        self.FWHMmax = float(value)
        print self.FWHMmax

    def batchpeakfit(self):

        self.roi = np.arange(self.pos1,self.pos2+1)
        self.pbutton_fit.setEnabled(False)
        
        self.PeakFitData = FitData(self.data,self.roi,self.Area,self.Areamin,self.Areamax,self.Pos,self.Posmin,self.Posmax,self.FWHM,self.FWHMmin,self.FWHMmax)
        self.PeakFitData.start()            
        self.PeakFitData.progress_fit.connect(self.progressbar_fit.setValue)
        self.PeakFitData.fitdone.connect(self.updatefitdata)

    def updatefitdata(self):
        
        self.res = self.PeakFitData.res #### need to think about this
        self.pbutton_fit.setEnabled(True)
        try:
            self.update()
        except:
            pass
        self.ChooseRes.setEnabled(True)
        self.pbutton_expfit.setEnabled(True)
        
    def stopfit(self):
        self.PeakFitData.terminate()
        self.progressbar_fit.setValue(0)
        self.pbutton_fit.setEnabled(True)
        
    def plot_fit_results(self,ind):
        
        self.mapper.axes.clear() # not fast
        
        if ind == 0:
            self.map_data = self.mapper.axes.imshow(self.res['Phase'],cmap='jet')
            title = "Phase distribution map"
            self.mapper.axes.set_title(title)
        elif ind == 1:
            self.map_data = self.mapper.axes.imshow(self.res['Position'],cmap='jet')
            title = "Peak position map"
            self.mapper.axes.set_title(title)        
        elif ind == 2:
            self.map_data = self.mapper.axes.imshow(self.res['FWHM'],cmap='jet')
            title = "FWHM map"
            self.mapper.axes.set_title(title)  
        elif ind == 3:
            self.map_data = self.mapper.axes.imshow(self.res['Background1'],cmap='jet')
            title = "Background 1 map"
            self.mapper.axes.set_title(title)  
        elif ind == 4:
            self.map_data = self.mapper.axes.imshow(self.res['Background2'],cmap='jet')
            title = "Background 2 map"
            self.mapper.axes.set_title(title)  
        
        try:
            self.cb.remove()
        except:
            pass
        self.cb = self.mapper.fig.colorbar(self.map_data)
        self.mapper.draw()                   
        
        
    def selCMin(self,value):
        self.CMin = float(value)
        self.CMinSel.setText(str(self.CMin))
        self.cb.set_clim(self.CMin,self.CMax)
        self.mapper.draw()
        
    def selCMax(self,value):
        self.CMax = float(value)
        self.CMaxSel.setText(str(self.CMax))        
        self.cb.set_clim(self.CMin,self.CMax)
        self.mapper.draw()        
        
    def savefitresults(self):
        
        if len(self.hdf_fileName)>0 and len(self.imoi)>0:
            
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_fit_phase.hdf5" %(s)
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.res['Phase'])
            h5f.create_dataset('Type', data='Phase')            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_fit_position.hdf5" %(s)
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.res['Position'])
            h5f.create_dataset('Type', data='Position')            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)                
            
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_fit_FWHM.hdf5" %(s)
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.res['FWHM'])
            h5f.create_dataset('Type', data='FWHM')            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)                            

            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_fit_Background1.hdf5" %(s)
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.res['Background1'])
            h5f.create_dataset('Type', data='Background1')            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_fit_Background2.hdf5" %(s)
            print sn

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.res['Background2'])
            h5f.create_dataset('Type', data='Background2')            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
        else:
            print "Something is wrong with the data"        
        
        
    ####################### ABS/PHC-CT data #######################
    
    def read_dark(self):
        
        dfn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open dark current image', "", "*.edf")
        self.dfn = dfn
        
        if len(self.dfn)>0:
            
            self.pbutton_darkpath.setText(self.dfn)
        
            self.dark = fabio.open(self.dfn)
            nd = self.dark.nframes
            dark_im = np.zeros((self.dark.dim2,self.dark.dim1))
            for ii in range(0,nd):
                dark_im = dark_im + self.dark.getframe(ii).data
            self.dark_im = dark_im/nd
            self.dark_im = np.array(self.dark_im, dtype = float)
            print 'Dark current image into memory'

            self.mapper.axes.clear() # not fast
            self.mapper.axes.imshow(self.dark_im,cmap='jet')
            title = 'Dark current'
            self.mapper.axes.set_title(title, fontstyle='italic')
            self.mapper.show()
            self.mapper.draw()    
            
    def read_flat(self):
        
        ffn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open flat field image', "", "*.edf")
        self.ffn = ffn
        
        if len(self.ffn)>0:
    
            self.pbutton_flatpath.setText(self.ffn)
            
            self.flat = fabio.open(self.ffn)
            nd = self.flat.nframes
            flat_im = np.zeros((self.flat.dim2,self.flat.dim1))
            for ii in range(0,nd):
                flat_im = flat_im + self.flat.getframe(ii).data
            self.flat_im = flat_im/nd
            self.flat_im = np.array(self.flat_im, dtype = float) - self.dark_im
            print 'Flat field image into memory'
            
            self.mapper.axes.clear() # not fast
            self.mapper.axes.imshow(self.flat_im,cmap='jet')
            title = 'Flat field'
            self.mapper.axes.set_title(title, fontstyle='italic')
            self.mapper.show()
            self.mapper.draw()                 

    def read_absdata(self):

        ifn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open absorption/phase contrast CT data', "", "*.edf")
        self.ifn = ifn
        
        if len(self.ifn)>0:
            
            print self.ifn
            self.pbutton_abspath.setText(self.ifn)            
            
            prefix = self.ifn.split("absct/")
            self.dataset = prefix[1]
            self.dataset = self.dataset.split("/")
            self.dataset = self.dataset[0]
            self.DatasetNameLabelA.setText(self.dataset)
            
            self.i = fabio.open(self.ifn)
            self.nd = self.i.nframes
            print 'Absorption/phase contrast CT data into memory'
            print '%d projections' %self.nd
            self.PlotRadio.setEnabled(True)
            
            self.imoi = self.i.getframe(1).data
            
            self.roixspinbox2.setMaximum(self.imoi.shape[0])
            self.roiyspinbox2.setMaximum(self.imoi.shape[1])
            self.roixspinbox1.setMaximum(self.imoi.shape[0])
            self.roiyspinbox1.setMaximum(self.imoi.shape[1])
            
            self.ofsspinbox.setMaximum(self.imoi.shape[0]-1)
            
            self.roixf = self.imoi.shape[0]
            self.roiyf = self.imoi.shape[1]
            
            self.roixspinbox2.setValue(self.roixf)
            self.roiyspinbox2.setValue(self.roiyf)            
            self.roixspinbox1.setValue(0)
            self.roiyspinbox1.setValue(0) 
            
            self.projspinbox.setMaximum(self.nd-1)
            self.zimspinbox.setMaximum(self.roiyf)


    def select_rowi(self,ind):
        self.roixi = ind
        print self.roixi
        
    def select_rowf(self,ind):
        self.roixf = ind
        print self.roixf

    def select_coli(self,ind):
        self.roiyi = ind
        print self.roiyi

    def select_colf(self,ind):
        self.roiyf = ind      
        print self.roiyf
        
    def select_proj(self,ind):
        self.projn = ind
        print self.projn        
        
    def plot_radiograph(self):

        try:
            self.roix = range(self.roixi,self.roixf)
            self.roiy = range(self.roiyi,self.roiyf)
            
            if self.nd>1:
                self.imoi = self.i.getframe(self.projn).data[self.roixi:self.roixf,self.roiyi:self.roiyf]
                print self.imoi.shape
                self.imoi = np.abs(-np.log((self.imoi-self.dark_im[self.roixi:self.roixf,self.roiyi:self.roiyf])/self.flat_im[self.roixi:self.roixf,self.roiyi:self.roiyf]))
                
                self.mapper.axes.clear() # not fast
                self.mapper.axes.imshow(self.imoi,cmap='jet')
                title = 'Radiograph'
                self.mapper.axes.set_title(title, fontstyle='italic')
                self.mapper.show()
                self.mapper.draw()         
                
                self.NormAbsVol.setEnabled(True)
        except:
            
            print 'Something is wrong with the roi or with the CT data'
            
    def selSavepath(self):
        savepath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory to save sinogram data"))
        self.savepathabs = savepath
        self.savedatapath.setText(self.savepathabs)  

    def norm_adata(self):

        if len(self.ffn)>0 and len(self.dfn)>0:
            self.i = []
            self.Normal = NormaliseABSCT(self.ifn,self.flat_im,self.dark_im,self.roixi,self.roixf,self.roiyi,self.roiyf)      
            self.Normal.start()
            
            self.Normal.progress_norm.connect(self.progressabsbarn.setValue)
            self.Normal.normdone.connect(self.ready)
                
    def ready(self):
        
        self.ReconstructAbsVol.setEnabled(True)

    def setscale(self,val):
        
        self.sc = float(val) #np.asarray(val , dtype="float64")
        print self.sc
        
    def changescantype(self,ind):
        if ind == 0:
            self.absscantype = "180"
            print "0 - 180"
#            self.labelst.setEnabled(False)
        elif ind == 1:
            self.absscantype = "360"
            print "0 - 360"
#            self.labelst.setEnabled(True)

    def select_ofs(self,ind):
        self.offset = ind
        print self.offset    
        
    def select_zim(self,ind):
        self.zim = ind
        print self.zim  
        
    def plot_stitched(self):
        
#        self.zim = self.i.getframe(0).data[:,self.roiyi:self.roiyf]
#        self.zim = np.round(self.zim.shape[0]/2)
        
        self.sino = np.zeros((len(range(self.roiyi,self.roiyf)),self.nd))
        print self.sino.shape
        for ii in range(0,self.nd):
            self.sino[:,ii] = (self.i.getframe(ii).data[self.zim,self.roiyi:self.roiyf]-self.dark_im[self.zim,self.roiyi:self.roiyf])/(self.flat_im[self.zim,self.roiyi:self.roiyf])

        self.sino = np.abs(-np.log((self.sino)) )                  
        self.sino = self.sino[:,50:self.sino.shape[1]-50]
        print self.sino.shape
        
        if self.absscantype == "360":
            
            s1 = self.sino[0:self.sino.shape[0]-self.offset,0:self.sino.shape[1]/2]
            print s1.shape
            s2 = np.flipud(self.sino[0:self.sino.shape[0]-self.offset,self.sino.shape[1]/2:self.sino.shape[1]])
            print s2.shape
            self.sino = np.concatenate((s1,s2),axis = 0)

        self.mapper.axes.clear() # not fast
        self.mapper.axes.imshow(self.sino,cmap='jet')
        title = 'Sinogram at the middle position of the volume'
        self.mapper.axes.set_title(title, fontstyle='italic')
        self.mapper.show()
        self.mapper.draw()           
        
        
    def reconabsvol(self):
        
        self.ReconABS = ReconABSCT(self.Normal.r,self.sc,self.savepathabs,self.dataset,self.absscantype,self.offset,self.sino)      
        self.ReconABS.start()
        
        self.ReconABS.progressabs.connect(self.progressabsbar.setValue)
#        self.ReconABS.progressabsrec.connect(self.showrec)
        self.ReconABS.recabsdone.connect(self.showrec)     
        self.pbutton_stop_abs.setEnabled(True)
        
    def showrec(self):
                
        self.mapper.axes.clear() # not fast
#        self.mapper.axes.imshow(self.ReconABS.bp[:,:,self.ReconABS.ch],cmap='jet')
#        title = 'Image %d out of %d' %(self.ReconABS.ch+1,self.ReconABS.total)
        self.mapper.axes.imshow(self.ReconABS.bp[:,:,0],cmap='jet')
        title = 'Image %d out of %d' %(1,self.ReconABS.total)        
        self.mapper.axes.set_title(title, fontstyle='italic')
        self.mapper.show()
        self.mapper.draw()             
        
        self.spinboxabs.setEnabled(True)
        self.spinboxabs.setMaximum(self.ReconABS.sn.shape[2]-1)
        self.sliderabs.setEnabled(True)
        self.sliderabs.setMaximum(self.ReconABS.sn.shape[2]-1)
        

    def changeImageN(self,value):
        
        self.spinboxabs.setValue(value)
        self.sliderabs.setValue(value)
            
        self.selectedChannels = value;
        self.mapper.axes.clear() # not fast
        self.mapper.axes.imshow(self.ReconABS.bp[:,:,self.selectedChannels],cmap='jet')
        title = "CT image no. %d" % (value)
        self.mapper.axes.set_title(title)
        self.mapper.show()
        self.mapper.draw()  

    def stopabsrec(self):
        
        try:
            self.ReconABS.terminate()
        except:
            print 'No process to terminate'
        self.progressabsbar.setValue(0)

        self.spinboxabs.setEnabled(False)
        self.sliderabs.setEnabled(False)


    def selABSCTdata(self):

        self.ABSdialog = FileDialog()
        if self.ABSdialog.exec_() == QtWidgets.QDialog.Accepted:
            self.abspaths = (self.ABSdialog.selectedFiles())
        
        for ii in range(0,len(self.abspaths)):
            oldpath = str(self.abspaths[ii])
            try:
                newpath = oldpath.split("/")
                self.datasetname = newpath[-1]
                
                try: 
                    newfile = self.datasetname.split(".edf")
                    newfile = newfile[0]
                    if len(self.datasetname)>0 and len(newfile)>0:
                        self.datalistabs.addItem(self.datasetname)    
                        
                        newpath = oldpath.split(self.datasetname)
                        newpath = newpath[0]
                        self.pathslistabs.append(newpath)  
                        
                except:
                    print('Something is wrong with the selected datasets')
            except:
                pass

    def remABSCTdata(self):

        for item in self.datalistabs.selectedItems():
            self.datalistabs.takeItem(self.datalistabs.row(item))
            
    def selFlatdata(self):

        self.Flatdialog = FileDialog()
        if self.Flatdialog.exec_() == QtWidgets.QDialog.Accepted:
            self.flatpaths = (self.Flatdialog.selectedFiles())
        
        for ii in range(0,len(self.flatpaths)):
            oldpath = str(self.flatpaths[ii])
            try:
                newpath = oldpath.split("/")
                self.datasetname = newpath[-1]
                
                try: 
                    newfile = self.datasetname.split(".edf")
                    newfile = newfile[0]
                    if len(self.datasetname)>0 and len(newfile)>0:
                        self.datalistflat.addItem(self.datasetname)    
                        
                        newpath = oldpath.split(self.datasetname)
                        newpath = newpath[0]
                        self.pathslistflat.append(newpath)  
                        
                except:
                    print('Something is wrong with the selected datasets')
            except:
                pass

    def remFlatdata(self):

        for item in self.datalistflat.selectedItems():
            self.datalistflat.takeItem(self.datalistflat.row(item))

    def selDarkdata(self):

        self.Darkdialog = FileDialog()
        if self.Darkdialog.exec_() == QtWidgets.QDialog.Accepted:
            self.darkpaths = (self.Darkdialog.selectedFiles())
        
        for ii in range(0,len(self.darkpaths)):
            oldpath = str(self.darkpaths[ii])
            try:
                newpath = oldpath.split("/")
                self.datasetname = newpath[-1]
                
                try: 
                    newfile = self.datasetname.split(".edf")
                    newfile = newfile[0]
                    if len(self.datasetname)>0 and len(newfile)>0:
                        self.datalistdark.addItem(self.datasetname)    
                        
                        newpath = oldpath.split(self.datasetname)
                        newpath = newpath[0]
                        self.pathslistdark.append(newpath)  
                        
                except:
                    print('Something is wrong with the selected datasets')
            except:
                pass


    def remDarkdata(self):

        for item in self.datalistdark.selectedItems():
            self.datalistdark.takeItem(self.datalistdark.row(item))


    def batchABSCTdata(self):

        # Need to disable some buttons
        
        datasetsabs = []
        datasetsflat = []
        datasetsdark = []
        for index in xrange(self.datalistabs.count()):
            datasetsabs.append(self.datalistabs.item(index).text())
            datasetsflat.append(self.datalistflat.item(index).text())
            datasetsdark.append(self.datalistdark.item(index).text())
            
        self.datasetsabs = datasetsabs
        self.datasetsflat = datasetsflat
        self.datasetsdark = datasetsdark
#        print self.datasets[0]
        
        self.BatchProcABS = []       
        self.Normal = []
        self.progressbarbatch = []
        self.labels = []
        
        for kk in range(0,len(self.datasetsabs)):
            
            self.prefix = self.datasetsabs[kk]

            filename = self.prefix.split(".edf")
            self.dataset = filename[0]
                    
#            output = '%s/%s_processed.hdf5' %(self.savepathabs,self.dataset)
#            print output

#            print 'self.pathslistabs'
#            print self.pathslistabs
#            print 'self.datasetsabs'
#            print self.datasetsabs
#            
#            print 'self.pathslistflat'
#            print self.pathslistflat
#            print 'self.datasetsflat'
#            print self.datasetsflat
#            
#            print 'self.pathslistdark'
#            print self.pathslistdark
#            print 'self.datasetsdark'
#            print self.datasetsdark            
            
            ####### For ABS/PHC-CT data #######
            self.ifn_path = '%s%s' %(self.pathslistabs[kk],self.datasetsabs[kk])
            print self.ifn_path
            
            ####### For flat field #######
            self.flat_im_path = '%s%s' %(self.pathslistflat[kk],self.datasetsflat[kk])
            print self.flat_im_path            
            self.flat = fabio.open(self.flat_im_path)
            nd = self.flat.nframes
            flat_im = np.zeros((self.flat.dim2,self.flat.dim1))
            for ii in range(0,nd):
                flat_im = flat_im + self.flat.getframe(ii).data
            self.flat_im = flat_im/nd
            self.flat_im = np.array(self.flat_im, dtype = float) - self.dark_im
            print 'Flat field image into memory'
            
            
            ####### For dark current #######
            self.dark_im_path = '%s%s' %(self.pathslistdark[kk],self.datasetsdark[kk])
            print self.dark_im_path                
            self.dark = fabio.open(self.dark_im_path)
            nd = self.dark.nframes
            dark_im = np.zeros((self.dark.dim2,self.dark.dim1))
            for ii in range(0,nd):
                dark_im = dark_im + self.dark.getframe(ii).data
            self.dark_im = dark_im/nd
            self.dark_im = np.array(self.dark_im, dtype = float)
            print 'Dark current image into memory'
            
            self.BatchProcABS.append(BatchProcessingABS(self.ifn_path,self.flat_im,self.dark_im,self.roixi,self.roixf,self.roiyi,self.roiyf,self.sc,self.savepathabs,self.dataset,self.absscantype,self.offset,self.sino))

        self.procnum = 0
        self.BatchProcABS[0].start()
        self.alldatasets = len(self.BatchProcABS)
        
        for ii in range(0,len(self.datasetsabs)):
            self.BatchProcABS[ii].batchrecsignal.connect(self.newrec)
        
    def newrec(self):
            
        self.procnum = self.procnum + 1
        if self.procnum<self.alldatasets:
            self.BatchProcABS[self.procnum].start()
        else:
            print 'All done'
        
        
        
        
        
                        
#            self.progressbarbatch.append(QtWidgets.QProgressBar(self))
#            self.tab1.layout.addWidget(self.progressbarbatch[ii],6+ii,5)
#            
#            self.labels.append(QtWidgets.QLabel(self))
#            self.labels[ii].setText(self.dataset)
#            self.tab1.layout.addWidget(self.labels[ii],6+ii,6)
#
#            self.BatchProc[ii].start()                
#            self.BatchProc[ii].progress.connect(self.progressbarbatch[ii].setValue)
            
            































        
        
        

    def loadabsct(self):
        h5_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open micro-CT data', "", "*.hdf5")

        print h5_fileName
        with h5py.File(h5_fileName,'r') as f:
            self.bpa = f['/bpa'][:]
        self.bpa = np.array(self.bpa)
        self.bpa = np.where(self.bpa<0,0,self.bpa)
        try:
            self.zaxis = np.arange(1,self.bpa.shape[2]+1)
        except:
            self.zaxis = 1
        print self.bpa.shape

        dim = self.bpa.shape
        if len(dim) == 3:
            self.slider.setMaximum(dim[2]-1)
            self.spinbox.setMaximum(dim[2]-1)
            self.abscorim.setMaximum(dim[2]-1)
        
    def fileOpen(self):
        self.hdf_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open XRD-CT data', "", "*.hdf5 *.h5")
        self.sinos = np.zeros(())
        self.bp = np.zeros(())
        
        if len(self.hdf_fileName)>0:
            self.loadxrdct()
       
            datasetname = self.hdf_fileName.split("/")
            self.datasetname = datasetname[-1]
            self.DatasetNameLabel.setText(self.datasetname)
    
            
            self.ReconstructVol.setEnabled(True)
            self.ProcessVol.setEnabled(True)
            self.ChooseXAxis.setEnabled(True)
    
            self.crspinbox1.setMaximum(self.data.shape[2])
            self.crspinbox2.setMaximum(self.data.shape[2])
            self.crspinbox3.setMaximum(self.data.shape[2])
            self.crspinbox4.setMaximum(self.data.shape[2])        
    
            self.slider_xrd.setMaximum(self.data.shape[2])        
            self.spinbox_xrd.setMaximum(self.data.shape[2])        
    
            try:
                self.cb.remove()
                self.histogramCurve.pop(0).remove()
                self.activeCurve.pop(0).remove()
                self.vCurve.remove()
            except:
                pass
    
            self.explore()
        
    def loadxrdct(self):
        if len(self.hdf_fileName)>0:
            with h5py.File(self.hdf_fileName,'r') as f:
                try:
                    self.data = f['/data'][:]
                except:
                    try:
                        self.sinos = f['/Sinograms'][:]
                        self.bp = f['/Reconstructed'][:]
                        self.data = self.bp
                        self.ChooseData.setCurrentIndex(1)
                        self.ReconstructVol.setEnabled(True)
                        self.ChooseData.setEnabled(True)
                        self.q = f['/q'][:]
                    except:
                        try:
                            self.data = f['/I'][:]
#                            self.data = np.transpose(self.data,(1,0,2))
                            self.tth = f['/twotheta'][:]
                            self.xaxis = self.tth
                            self.sinos = self.data
                        except:
                            pass
    
                # Just for battery stuff
                try:
                    self.sinos = f['/sinograms_coarse'][:];self.sinos = np.transpose(self.sinos,(2,1,0))
                    self.bp = f['/reconstructed_coarse'][:];self.bp = np.transpose(self.bp,(2,1,0))
                    self.q = f['/q'][:]
                    self.data = self.bp
                    self.ChooseData.setCurrentIndex(1)
                    self.ReconstructVol.setEnabled(True)
                    self.ChooseData.setEnabled(True)
                except:
                    pass

                try:
                    self.tth = f['/twotheta'][:]
                    self.q = f['/q'][:]
                    self.d = f['/d'][:]
                    self.na = f['/slow_axis_steps'].value
                    self.nt = f['/fast_axis_steps'].value                      
                    self.scantype = f['/scantype'].value
                    self.omega = f['/omega'][:]
                    try:
                        self.y = f['/translations'][:]
                    except:
                        try:
                            self.y = f['/y'][:]
                        except:
                            pass
                    self.dio = f['/diode'][:]
                    self.etime = f['/exptime'][:]
                    self.dio = self.dio/np.max(self.dio)
                    self.etime = self.etime/np.max(self.etime)                    
                    self.xaxisd = self.d[::-1]
                    self.E = f['/energy'].value         
                    self.EnergySel.setText(str(self.E))
                except:
                    pass
		    
#        try:
#            if len(self.dio.shape)>0 and len(self.etime.shape)>0:
#    
#                for ii in range(0,self.data.shape[0]):
#                    self.data[ii,:] = self.data[ii,:]/self.dio[ii]
#    #                self.data[ii,:] = self.data[ii,:]/self.etime[ii]   
#                print 'Diode normalization done'
#        except:
#            print 'No diode values'
            
#        if self.scantype == 'zigzag':
#            self.y = np.reshape(self.y,(int(self.na),int(self.nt)))
#            self.y = np.transpose(self.y,(1,0))
#            self.y[:,0::2] = self.y[::-1,0::2]
#            print self.y.shape
        
        if self.scantype == 'zigzag' and len(self.data.shape)<3:
            self.data = np.reshape(self.data,(int(self.na),int(self.nt),self.data.shape[1]))
            self.data = np.transpose(self.data,(1,0,2))
            self.data[:,0::2,:] = self.data[::-1,0::2,:]
            self.sinos = self.data
            
        elif self.scantype == 'fast' and len(self.data.shape)<3:
                        
            cr = -100.0;

            in_y = np.array(np.where(self.y>cr)); 
            in_y = in_y[:,0];
		    
            an_step = self.omega[1] - self.omega[0];
            omega_roi = self.omega[in_y[0]::];
            in_a = np.array(np.where(omega_roi>(360-an_step)));
            in_a = in_a[:,0];
		    
            useful = in_y[0]+in_a[0]+1;
		    
            print useful
		    
		    #%
		    
		    
            xold = self.y[0:useful];
            yold = np.sum(self.data[0:useful,:],axis = 1);

            xnew = np.linspace(xold[0],xold[-1],len(xold));
            ynew = np.interp(xnew, xold, yold, left=0 , right=0)

   

            sn = np.reshape(ynew,(useful/self.na,self.na))
	#            plt.figure(5);plt.clf();plt.imshow(sn, cmap = 'jet');plt.show();
		    
            s1 = sn[0::2,:]
            s2 = sn[1::2,:]
		    
	#            snew = np.zeros((s1.shape[0]+s1.shape[0],s1.shape[1]))
	#            snew[0:s1.shape[0]] = s1
	#            snew[s1.shape[0]::] = s2[::-1,:]
		    
	#            plt.figure(6);plt.clf();plt.imshow(snew, cmap = 'jet');plt.show();
		    
		    
            snew = np.zeros((s1.shape[0]+s1.shape[0]-1,s1.shape[1],self.data.shape[1]))

            for ii in range(0,self.data.shape[1]):
		        
                yold = self.data[0:useful,ii];
                ynew = np.interp(xnew, xold, yold, left=0 , right=0)
		    
                sn = np.reshape(ynew,(useful/self.na,self.na))
                s1 = sn[0::2,:]
                s2 = sn[1::2,:]
		    
                snew[0:s1.shape[0]-1,:,ii] = s1[0:s1.shape[0]-1,:]
                snew[s1.shape[0]-1::,:,ii] = s2[::-1,:]
		    
            self.data = snew
            self.sinos = self.data
#        else:
#            self.sinos = self.data


		        
        if len(self.tth.shape)>0:
		        self.xaxis = self.tth
        else:                
		        self.xaxis = self.q
		        self.xaxislabel = 'Q'          
		        self.ChooseXAxis.setEnabled(False)
        print self.data.shape
	       

    def loadparfile(self):   
    
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Scan Parameters File', "", "*.json")
    
        try:
            if len(fn)>0:
                d = json.load(open(fn))
                self.E = d["Energy"];self.datatype=d["Datatype"];self.calibrant=d["CalibrantPath"];self.poniname=d["PoniPath"];self.maskname=d["MaskPath"];self.mask=d["Mask"];
                self.jsonname=d["JsonPath"];self.npt_rad=d["RadialPoints"];self.units=d["Units"];self.savepath=d["SavePath"];self.xrdctpath=d["XRDCTPath"];self.procunit=d["ProcessingUnit"];
                self.filt=d["Filter"];self.prc=d["TrimmedMean"];self.thres=d["Sigma"];self.scantype=d["ScanType"];self.parfile=d["ParFile"];
                self.nt=d["SlowAxisSteps"];self.na=d["FastAxisSteps"];
                self.dataset=d["Dataset"];self.prefix=d["Prefix"];#;self.omega=d["Omega"];self.trans=d["Translations"]
        		
                self.E = float(self.E);self.na = int(self.na);self.nt = int(self.nt);
        		# now need to display these things
    #                self.EnergySel.setText(str(self.E));
    
        except:
            pass


        
    def savexrdct(self):

        self.fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save XRD-CT data', "", "*.hdf5")
	
        if len(self.fn)>0:
    
            st = self.fn.split(".hdf5")
            if len(st)<2:
                self.fn = "%s.hdf5" %self.fn
                print self.fn

#            with open(self.fn, 'w') as outfile:  
#                json.dump(self.dict, outfile)

            h5f = h5py.File(self.fn, "w")
            h5f.create_dataset('Sinograms', data=self.sinos)
            h5f.create_dataset('twotheta', data=self.tth)
            h5f.create_dataset('q', data=self.q)
            h5f.create_dataset('d', data=self.d)
            
            dims = self.bp.shape
            if len(dims)>1:
                h5f.create_dataset('Reconstructed', data=self.bp)            
#            h5f.create_dataset('scantype', data=self.scantype)
            h5f.close()
        
            perm = 'chmod 777 %s' %self.fn
            os.system(perm)    
        

    def fileQuit(self):
#        plt.close('all')
        self.close()

    def closeEvent(self, ce):
#        plt.close('all')
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Data visualizer application""")
      
      
class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class MyCanvas(Canvas):
    #Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
#        self.axes.hold(False)

        self.compute_initial_figure()
        Canvas.__init__(self, fig)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
       
    def compute_initial_figure(self):
        pass
    

class FileDialog(QtWidgets.QFileDialog):
        def __init__(self, *args):
            QtWidgets.QFileDialog.__init__(self, *args)
            self.setOption(self.DontUseNativeDialog, True)
#            self.setFileMode(self.DirectoryOnly)
            for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                if isinstance(view.model(), QtWidgets.QFileSystemModel):
                    view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
                    
#def main():
#    qApp = QtWidgets.QApplication(sys.argv)
#    aw = ApplicationWindow()
#    aw.show()
#    sys.exit(qApp.exec_())
#    qApp.exec_()
#    
#if __name__ == "__main__":
#    main()
    
aw = ApplicationWindow()    
aw.show()
    
