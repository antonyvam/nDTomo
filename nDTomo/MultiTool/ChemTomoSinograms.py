# -*- coding: utf-8 -*-
"""
Chemical tomography sinogram data class for the MultiTool

@author: Antony
"""

from PyQt5.QtCore import pyqtSignal, QThread
from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round

class SinoProcessing(QThread):
    '''
    The SinoProcessing class allows for:
        a) Centering of the sinogram(s)
        b) Subtraction of background signal
        c) Normalisation of sinograms assuming constant summed intensity per projection
    '''
    
    snprocdone = pyqtSignal()
    progress_sino = pyqtSignal(int)
    
    def __init__(self,sinos,sinonorm,ofs,crsr,scantype):
        QThread.__init__(self)
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
            dpair = (mean(self.sinos_proc[0:self.ofs,ii,:],axis = 0) + mean(self.sinos_proc[self.sinos_proc.shape[0]-self.ofs:self.sinos_proc.shape[0],ii,:],axis = 0))/2
            self.sinos_proc[:,ii,:] = self.sinos_proc[:,ii,:] - dpair
            
    def scalesinos(self):
        
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
#                    col = self.sinos_proc[:,ii,ll]
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
#                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)
                sn[:,ii] = interp(xnew, xold, s[:,ii])
                
            if self.scantype == 'zigzag':
                sn = sn[:,0:sn.shape[1]-1]
            
        self.sinos_proc = sn
        self.snprocdone.emit()