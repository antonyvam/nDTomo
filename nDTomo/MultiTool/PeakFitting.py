# -*- coding: utf-8 -*-
"""

Single peak batch fitting class for the MultiTool

@author: A. Vamvakeros

"""

#%

from numpy import pi, argwhere, concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round
import scipy.optimize as sciopt
from PyQt5.QtCore import pyqtSignal, QThread

class FitData(QThread):
    
    '''

    Single peak batch fitting class   
    
    :data: the spectral/scattering data
    
    :roi: bin number of interest

    :Area: initial value for peak area

    :Areamin: minimum value for peak area

    :Areamax: maximum value for peak area

    :Pos: initial value for peak position

    :Posmin: minimum value for peak position

    :Posmax: maximum value for peak position

    :FWHM: initial value for peak full width at half maximum (FWHM)

    :FWHMmin: minimum value for peak FWHM

    :FWHMmax: maximum value for peak FWHM

    '''
    
    fitdone = pyqtSignal()
    progress_fit = pyqtSignal(int)
    
    def __init__(self,peaktype,data,roi,Area,Areamin,Areamax,Pos,Posmin,Posmax,FWHM,FWHMmin,FWHMmax):
        QThread.__init__(self)
        
        self.peaktype = peaktype
        self.fitdata = data    
        self.phase = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.cen = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.wid = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg1 = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg2 = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.fr = zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.xroi = roi
        self.Area = Area; self.Areamin = Areamin; self.Areamax = Areamax; 
        self.Pos = Pos; self.Posmin = Posmin; self.Posmax = Posmax; 
        self.FWHM = FWHM; self.FWHMmin = FWHMmin; self.FWHMmax = FWHMmax;
        
        
        im = sum(self.fitdata,axis = 2);
        im = im/max(im);
        im = where(im<0.15,0,im);
        msk = im; 
        msk = where(msk>0,1,0)
        
        for ii in range(0,self.fitdata.shape[2]):
            self.fitdata[:,:,ii] = self.fitdata[:,:,ii]*msk
            
        self.i, self.j = where(msk > 0)

    def run(self):
        
        """
        
        Initialise the single peak batch fitting process
        
        """  
        
        self.batchfit()
        
    def batchfit(self):
                
        mdp = mean(mean(self.fitdata, axis=0),axis = 0)
        y = mdp[self.xroi]
        
        if self.peaktype == "Pseudo-Voigt":
            x0 = array([float(self.Area), float(self.Pos), float(self.FWHM), 0., float(min(y)), 0.5], dtype=float)
        else:
            x0 = array([float(self.Area), float(self.Pos), float(self.FWHM), 0., float(min(y))], dtype=float)
#        x0 = np.array([1., np.mean(self.xroi), 1., 0., np.min(y)], dtype=float)
        
        if self.peaktype == "Pseudo-Voigt":
            param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-inf,-inf,0],[float(self.Areamax),float(self.Posmax),float(self.FWHMmax),inf,inf,1])
        else:
            param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-inf,-inf],[float(self.Areamax),float(self.Posmax),float(self.FWHMmax),inf,inf])
        
#        param_bounds=([-1,np.mean(self.xroi)-5.,0,-np.inf,-np.inf],[1E3,np.mean(self.xroi)+5,1E3,np.inf,np.inf])
          
#        for ii in range(0,self.fitdata.shape[0]):
#            for jj in range(0,self.fitdata.shape[1]):

        for ind in arange(0,len(self.i)):
                
                ii = self.i[ind]
                jj = self.j[ind]
                
                dp = self.fitdata[ii,jj,:]
                           
                try:
                    
                    print('Row %d column %d '%(ii, jj))
                    
                    if self.peaktype == "Gaussian":
                        params, params_covariance = sciopt.curve_fit(self.gmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Lorentzian":
                        params, params_covariance = sciopt.curve_fit(self.lmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Pseudo-Voigt":
                        params, params_covariance = sciopt.curve_fit(self.pvmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    
                    self.phase[ii,jj] = params[0]
                    self.cen[ii,jj] = params[1]
                    self.wid[ii,jj] = params[2]         
                    self.bkg1[ii,jj] = params[3]   
                    self.bkg2[ii,jj] = params[4]      
                    
                    if self.peaktype == "Pseudo-Voigt":
                        self.fr[ii,jj] = params[5]  
           
                except: 
                    self.phase[ii,jj] = 0
                    self.cen[ii,jj] = (param_bounds[0][1] + param_bounds[1][1])/2
                    self.wid[ii,jj] = (param_bounds[0][2] + param_bounds[1][2])/2     
                    self.bkg1[ii,jj] = 0
                    self.bkg2[ii,jj] = 0

                    if self.peaktype == "Pseudo-Voigt":
                        self.fr[ii,jj] = 1
                        
#                v = (100.*(ii+1))/(self.fitdata.shape[0])
                v = (100.*(ind+1))/(len(self.i))
                self.progress_fit.emit(v)
                
        self.phase = where(self.phase<0,0,self.phase)
        
        if self.peaktype == "Pseudo-Voigt":
            self.res = {'Phase':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2, 'Fraction':self.fr}
        else:
            self.res = {'Phase':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2}
        
        self.fitdone.emit()

    def gmodel(self, x, A, m, w, a, b):
        
        """
        
        Gaussian model with linear background: (A/(sqrt(2*pi)*w) )* exp( - (x-m)**2 / (2*w**2)) + a*x + b
        
        """
        return (A/(sqrt(2*pi)*w) )* exp( - (x-m)**2 / (2*w**2)) + a*x + b
    
    def lmodel(self, x, A, m, w, a, b):
        
        """
        
        Lorentzian model with linear background: (A/(1 + ((1.0*x-m)/w)**2)) / (pi*w) + a*x + b   
        
        """
        return (A/(1 + ((x-m)/w)**2)) / (pi*w) + a*x + b    

    def pvmodel(self, x, A, m, w, a, b, fr):
        
        """
        
        pseudo-Voigt model with linear background: ((1-fr)*gaumodel(x, A, m, s) + fr*lormodel(x, A, m, s))
        
        """
        return ((1-fr)*(A/(sqrt(2*pi)*w) )* exp( - (x-m)**2 / (2*w**2)) + fr*(A/(1 + ((x-m)/w)**2)) / (pi*w) + a*x + b)