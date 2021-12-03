# -*- coding: utf-8 -*-
"""
Chemical tomography data reconstruction class for the MultiTool

@author: A. Vamvakeros
"""

from PyQt5.QtCore import pyqtSignal, QThread
from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
from skimage.transform import iradon
import os, h5py

try:
    from  silx.opencl.backprojection import Backprojection as fbp
except:
    print("Silx is not installed or there is a problem with pyopencl")
    
class ReconstructData(QThread):

    '''
    The chemical tomography data can be reconstructed with either the skimage or silx.
    In both cases, the filterec back projection algorithm is emploed to reconstruct the images.
    
        :sinos: sinogram data volume
        
    '''
    
    recdone = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self,sinos):
        QThread.__init__(self)
        self.sinos = sinos
        
    def run(self):

        """
        
        Initialise the data reconstruction process
        
        """   
        
        npr = self.sinos.shape[1]
        self.theta = arange(0,180-180./(npr-1),180./(npr-1))
        self.sinos = self.sinos[0:self.sinos.shape[0],0:len(self.theta),:]
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
        self.remring()
        
        self.recdone.emit()
        
    def rec_skimage(self):

        """
        
        Reconstruct the images using the filtered back projection algorithm from the skimage package
        
        """
        
        self.bp = zeros((self.sinos.shape[0],self.sinos.shape[0],self.sinos.shape[2]))
        
        ist = 0
        ifi = self.sinos.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sinos[:,0:len(self.theta),ii], theta=self.theta, output_size=self.sinos.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
    def rec_silx(self):

        """
        
        Reconstruct the images using the filtered back projection algorithm from the silx package using GPU
        
        """
        
        self.bp = zeros((self.sinos.shape[0],self.sinos.shape[0],self.sinos.shape[2]))
        t = fbp(sino_shape=(self.sinos.shape[1],self.sinos.shape[0]),devicetype='CPU')
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sinos.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=transpose(self.sinos[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
    def remring(self):
        
        """
        
        Remove recontruction ring
        
        """
        
        sz = floor(self.bp.shape[0])
        x = arange(0,sz)
        x = tile(x,(int(sz),1))
        y = swapaxes(x,0,1)
        
        xc = round(sz/2)
        yc = round(sz/2)
        
        r = sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = where(r>0.98*sz/2,0,self.bp[:,:,ii])



class BatchProcessing(QThread):
    
    '''
    
    Class for batch reconstruction of multiple chemical tomography data  
    
    :sinos_proc: sinogram data volume
        
    :sinonorm: normalisation of the sinogram data volume, options are 0 or 1.
    
    :ofs: number of voxels (from each side of the sinograms) to be used for background subtraction
    
    :crsr: maximum number of voxels to use for the sinogram centering process
    
    :scantype: type of scan, options are: 'Zigzag', 'ContRot' and 'Interlaced'
    
    :output: output file containing the results

    :tth: 2theta (1d array)
        
    :d: d spacing (1d array)
        
    :q: q spacing (1d array)
    
    '''
    
    snprocdone = pyqtSignal()
    progress_sino = pyqtSignal(int)
    recdone = pyqtSignal()
    progress = pyqtSignal(int)
    
    def __init__(self,sinos,sinonorm,ofs,crsr,scantype,output,tth,q,d):
        QThread.__init__(self)
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
        
        """
        
        Initialise the batch data reconstruction process
        
        """
        
        if self.sinonorm == 1:
            self.scalesinos()   
        
        if self.ofs>0:
            self.airrem()

        if self.crsr>0:       
            self.sinocentering()
        
        npr = self.sinos_proc.shape[1]
        self.theta = arange(0,180-180./(npr-1),180./(npr-1))
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
        
        """
        
        Method for subtracting the backgroung signal from the sinograms
        
        """           
        
        for ii in range(0,self.sinos_proc.shape[1]):
            dpair = (mean(self.sinos_proc[0:self.ofs,ii,:],axis = 0) + mean(self.sinos_proc[self.sinos_proc.shape[0]-self.ofs:self.sinos_proc.shape[0],ii,:],axis = 0))/2
            self.sinos_proc[:,ii,:] = self.sinos_proc[:,ii,:] - dpair
            
    def scalesinos(self):
        
        """
        
        Method for normalising the sinograms. It assumes that the total scattering intensity per projection is constant.
        
        """  
        
        ss = sum(self.sinos_proc,axis = 2)
        scint = zeros((self.sinos_proc.shape[1]))
        # Summed scattering intensity per linescan
        for ii in range(0,self.sinos_proc.shape[1]):
            scint[ii] = sum(ss[:,ii])
        # Scale factors
        sf = scint/max(scint)
        
        # Normalise the sinogram data    
        for jj in range(0,self.sinos_proc.shape[1]):
            self.sinos_proc[:,jj,:] = self.sinos_proc[:,jj,:]/sf[jj] 
        
    def sinocentering(self):
                
        """
        
        Method for centering sinograms by flipping the projection at 180 deg and comparing it with the one at 0 deg
        
        """   
        
        di = self.sinos_proc.shape
        if len(di)>2:
            s = sum(self.sinos_proc, axis = 2)
        
        cr =  arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.1)
    #    nomega = s.shape[1] - 1
        
        xold = arange(0,s.shape[0])
        
        st = []; ind = [];
        
        
        for kk in range(0,len(cr)):
            
            xnew = cr[kk] + arange(-ceil(s.shape[0]/2),ceil(s.shape[0]/2)-1)
            sn = zeros((len(xnew),s.shape[1]))
            
            
            for ii in range(0,s.shape[1]):
                
                sn[:,ii] = interp(xnew, xold, s[:,ii])
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)

            re = sn[::-1,-1]
    #        re = sn[:,:1]
            st.append((std((sn[:,0]-re)))); ind.append(kk)
    #        plt.figure(1);plt.clf();plt.plot(sn[:,0]);plt.plot(re);plt.pause(0.001);
    
        m = argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + arange(-ceil(s.shape[0]/2),ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = zeros((len(xnew),self.sinos_proc.shape[1],self.sinos_proc.shape[2]))  
            for ll in range(0,self.sinos_proc.shape[2]):
                for ii in range(0,self.sinos_proc.shape[1]):
                    sn[:,ii,ll] = interp(xnew, xold, self.sinos_proc[:,ii,ll])    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
                v = (100.*(mm+1))/self.sinos_proc.shape[2]
                mm = mm + 1
                self.progress_sino.emit(v)
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1,:]
                
#                print ll
                
        elif len(di)==2:
            
            sn = zeros((len(xnew),s.shape[1]))    
            for ii in range(0,s.shape[1]):
                sn[:,ii] = interp(xnew, xold, s[:,ii], left=0 , right=0)
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1]
            
        self.sinos_proc = sn
        self.snprocdone.emit()
        
    def rec_skimage(self):

        self.bp = zeros((self.sinos_proc.shape[0],self.sinos_proc.shape[0],self.sinos_proc.shape[2]))
        
        ist = 0
        ifi = self.sinos_proc.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sinos_proc[:,0:len(self.theta),ii], theta=self.theta, output_size=self.sinos_proc.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
        
    def rec_silx(self):

        """
        
        Reconstruct the images using the filtered back projection algorithm from the silx package using GPU
        
        """        
        
        self.bp = zeros((self.sinos_proc.shape[0],self.sinos_proc.shape[0],self.sinos_proc.shape[2]))
        t = fbp(sino_shape=(self.sinos_proc.shape[1],self.sinos_proc.shape[0]),devicetype='CPU')
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sinos_proc.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=transpose(self.sinos_proc[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progress.emit(v)
            
    def remring(self):

        """
        
        Remove recontruction ring
        
        """
        
        sz = floor(self.bp.shape[0])
        x = arange(0,sz)
        x = tile(x,(int(sz),1))
        y = swapaxes(x,0,1)
        
        xc = round(sz/2)
        yc = round(sz/2)
        
        r = sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.bp.shape
        if len(dim)==2:
            self.bp = where(r>0.98*sz/2,0,self.bp)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.bp[:,:,ii] = where(r>0.98*sz/2,0,self.bp[:,:,ii])
        try:
            self.bp = where(self.bp<0,0,self.bp)
        except:
            print("No negative values in the reconstructed data")
        
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
        
            print('Dataset %s has been processed and saved' %(self.output))
            
            perm = 'chmod 777 %s' %self.output
            try:
                os.system(perm)  
            except:
                'Wrote the data but problem with permissions'