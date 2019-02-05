# -*- coding: utf-8 -*-
"""
Absorption-contrast tomography data reconstruction class for the MultiTool

@author: A. Vamvakeros
"""

from PyQt5.QtCore import pyqtSignal, QThread
from skimage.transform import iradon
from os import system
from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
try:
    from scipy.misc import imresize
except:
    print "Cannot import imresize and/or imrotate"
try:
    from  silx.opencl.backprojection import Backprojection as fbp
except:
    print "Silx is not installed or there is a problem with pyopencl"
import h5py

class ReconABSCT(QThread):
    
    '''
    
    The absorption-contrast tomography data can be reconstructed with either the skimage or silx.
    In both cases, the filterec back projection algorithm is emploed to reconstruct the images. 
    
    :sinos: sinogram data volume
        
    :sc: scale factor used to resize the sinograms (max value of 1)
    
    :savepathabs: directory to save the reconstructed absorption-contrast CT data
    
    :dataset: dataset name
    
    :absscantype: type of scan, options are: '180' or '360'
    
    :offset: maximum number of pixels to use for the sinogram centering process
    
    '''
    
    recabsdone = pyqtSignal()
    progressabs = pyqtSignal(int)
    progressabsrec = pyqtSignal()
    
    def __init__(self,sinos,sc,savepathabs,dataset,absscantype,offset):
        QThread.__init__(self)
        self.sinos = transpose(sinos,(1,2,0))
#        self.sinos = self.sinos[:,50:self.sinos.shape[1]-50,:] # The GUI should ask how many extra images we are collecting, in this case it is 100 images
        print self.sinos.shape
        self.sc = sc
        print self.sc
        self.savepathabs = savepathabs
        self.dataset = dataset
        self.absscantype = absscantype
        self.offset = offset
                
        if self.absscantype == "360":
            
            self.sn = zeros((self.sinos.shape[0],self.sinos.shape[1],self.sinos.shape[2]))
            print self.sn.shape
            
            for ii in range(0,self.sinos.shape[2]):
                s1 = self.sinos[0:self.sinos.shape[0]-self.offset,0:self.sinos.shape[1]/2,ii]
                s2 = flipud(self.sinos[0:self.sinos.shape[0]-self.offset,self.sinos.shape[1]/2:self.sinos.shape[1],ii])
                self.sn[:,:,ii] = concatenate((s1,s2),axis = 0)       
                
                print 'Stitching sinogram %d out of %d' %(ii+1,self.sinos.shape[2])
            self.sinos = self.sn
        
        
        if self.sc<1:
            sn =  imresize(self.sinos[:,:,0], self.sc, 'bilinear')        
            self.sn = zeros((sn.shape[0],sn.shape[1],self.sinos.shape[2]))
            
            for ii in range(0,self.sinos.shape[2]):
                self.sn[:,:,ii] = imresize(self.sinos[:,:,ii], self.sc, 'bilinear')
                print 'Resizing image %d out of %d' %(ii,self.sinos.shape[2])
                
        else:
            self.sn = self.sinos
            
        self.sinos = []
        self.crsr = 0.2*self.sn.shape[0]
        self.ofs = 2
        
    def run(self):
        
        """
        
        Initialise the data reconstruction process
        
        """   
        
        self.airrem()
        self.sinocentering()

        npr = self.sn.shape[1]
#        self.theta = np.arange(0,180-180./(npr-1),180./(npr-1))
        
        self.theta = arange(0,180,180./(npr))
        print self.theta.shape
#        self.sinos = self.sinos[0:self.sinos.shape[0],0:len(self.theta),:]
        self.total = self.sn.shape[2]
        
        self.bp = zeros((self.sn.shape[0],self.sn.shape[0],self.sn.shape[2]))
        
        try:
            self.rec_silx()
        except:
            self.rec_skimage()
        
        self.remring()
        self.bp = where(self.bp<0,0,self.bp)
        
        self.savedata()
        self.recabsdone.emit()

    def airrem(self):
        
        """
        
        Method for subtracting the backgroung signal from the sinograms
        
        """               
        
        for ii in range(0,self.sn.shape[1]):
            dpair = (mean(self.sn[0:self.ofs,ii,:],axis = 0) + mean(self.sn[self.sn.shape[0]-self.ofs:self.sn.shape[0],ii,:],axis = 0))/2
            self.sn[:,ii,:] = self.sn[:,ii,:] - dpair
        print "Air background subtracted"
            
    def sinocentering(self):
        
        """
        
        Method for centering sinograms by flipping the projection at 180 deg and comparing it with the one at 0 deg
        
        """        
        
        di = self.sn.shape
        if len(di)>2:
            s = sum(self.sn, axis = 2)
        
        cr =  arange(s.shape[0]/2 - self.crsr, s.shape[0]/2 + self.crsr, 0.5)
    #    nomega = s.shape[1] - 1
        
        xold = arange(0,s.shape[0])
        
        st = []; ind = [];
        
        print "Trying to centre the global sinogram..."
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
    
            print 'Offset no. %d out of %d' %(kk,len(cr))
    
        m = argmin(st)
        print(cr[m])
        
        mm = 0
        xnew = cr[m] + arange(-ceil(s.shape[0]/2),ceil(s.shape[0]/2)-1)
        if len(di)>2:
            sn = zeros((len(xnew),self.sn.shape[1],self.sn.shape[2]))  
            for ll in range(0,self.sn.shape[2]):
                for ii in range(0,self.sn.shape[1]):
                    col = self.sn[:,ii,ll]
                    sn[:,ii,ll] = interp(xnew, xold, col)    
#                    sn[:,ii,ll] = np.interp(xnew, xold, self.sinos_proc[:,ii,ll], left=0 , right=0) 
                    
#                v = (100.*(mm+1))/self.sn.shape[2]
                mm = mm + 1
                print 'Centering sinogram %d out of %d' %(ll,self.sn.shape[2])
                
        self.sn = sn
        print "All sinograms centered"
        
    def rec_skimage(self):
        
        """
        
        Reconstruct the images using the filtered back projection algorithm from the skimage package
        
        """
        
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = iradon(self.sn[:,:,ii], theta=self.theta, output_size=self.sn.shape[0], circle=True)
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
        
    def rec_silx(self):
        
        
        """
        
        Reconstruct the images using the filtered back projection algorithm from the silx package using GPU
        
        """
          
#        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='CPU')
        
        t = fbp(sino_shape=(self.sn.shape[1],self.sn.shape[0]),devicetype='GPU')
        
#            print self.bp.shape
#            start=time.time()
        ist = 0
        ifi = self.sn.shape[2]
        for ii in range(ist,ifi):
            self.bp[:,:,ii] = t.filtered_backprojection(sino=transpose(self.sn[:,:,ii],(1,0)))
            v = (100.*(ii-ist+1))/(ifi-ist)
            self.progressabs.emit(v)
            
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
                
    def remring_single(self):
        
        sz = floor(self.bpoi.shape[0])
        x = arange(0,sz)
        x = tile(x,(int(sz),1))
        y = swapaxes(x,0,1)
        
        xc = round(sz/2)
        yc = round(sz/2)
        
        r = sqrt(((x-xc)**2 + (y-yc)**2));
        
        self.bpoi = where(r>0.98*sz/2,0,self.bpoi)


    def savedata(self):
        
        """
        
        Save the reconstructed 3D data in a single .hdf5 file
        
        """

        self.output = '%s/%s_rec.hdf5' %(self.savepathabs,self.dataset)
        
        if len(self.output)>0:
    
            h5f = h5py.File(self.output, "w")
            h5f.create_dataset('bpa', data=self.bp)
            
            h5f.close()
        
            print 'Dataset %s has been processed and saved' %(self.output)
            
            perm = 'chmod 777 %s' %self.output
            try:
                system(perm)  
            except:
                'Wrote the data but problem with permissions'
                
                