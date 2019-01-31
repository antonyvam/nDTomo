# -*- coding: utf-8 -*-
"""
Single peak batch fitting class for the MultiTool

@author: Antony
"""

#%

from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
import scipy.optimize as sciopt
from PyQt5.QtCore import pyqtSignal, QThread

class FitData(QThread):
    
    '''
    Single peak batch fitting class    
    '''
    
    fitdone = pyqtSignal()
    progress_fit = pyqtSignal(int)
    
    def __init__(self,data,roi,Area,Areamin,Areamax,Pos,Posmin,Posmax,FWHM,FWHMmin,FWHMmax):
        QThread.__init__(self)
        self.fitdata = data    
        self.phase = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.cen = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.wid = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg1 = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg2 = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.xroi = roi
        self.Area = Area; self.Areamin = Areamin; self.Areamax = Areamax; 
        self.Pos = Pos; self.Posmin = Posmin; self.Posmax = Posmax; 
        self.FWHM = FWHM; self.FWHMmin = FWHMmin; self.FWHMmax = FWHMmax;
        
        
        im = sum(self.fitdata,axis = 2);
        im = im/max(im);
        im = where(im<0.1,0,im);
        msk = im; 
        msk = where(msk>0,1,0)
        
        for ii in range(0,self.fitdata.shape[2]):
            self.fitdata[:,:,ii] = self.fitdata[:,:,ii]*msk

    def run(self):
        self.batchfit()
        
    def batchfit(self):
                
        mdp = mean(mean(self.fitdata, axis=0),axis = 0)
        y = mdp[self.xroi]
        x0 = array([float(self.Area), float(self.Pos), float(self.FWHM), 0., float(min(y))], dtype=float)
#        x0 = np.array([1., np.mean(self.xroi), 1., 0., np.min(y)], dtype=float)
        param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-inf,-inf],[float(self.Areamax),float(self.Posmax),float(self.FWHMmax),inf,inf])
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
                
                self.phase = where(self.phase<0,0,self.phase)
                self.res = {'Phase':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2}
        
        self.fitdone.emit()

    def gmodel2(self, x, A, m, w, a, b):
        return A * exp( - (x-m)**2/w) + a*x + b