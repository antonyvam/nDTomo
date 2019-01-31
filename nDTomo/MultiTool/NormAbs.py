# -*- coding: utf-8 -*-
"""
Radiograph normalisation class for the MultiTool

@author: Antony
"""

from PyQt5 import QtCore
import fabio
from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round

class NormaliseABSCT(QtCore.QThread):
    '''
    The NormaliseABSCT class allows for normalisation of an absortion-contrast 
    '''    
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
        
        self.r = zeros((len(self.roix),len(self.roiy),len(self.roip)))
        
        kk = 0
        for ii in self.roip:
            im = self.i.getframe(ii).data[self.roixi:self.roixf,self.roiyi:self.roiyf]
            self.r[:,:,kk] = abs(-log((im-self.dark_im[self.roixi:self.roixf,self.roiyi:self.roiyf])/self.flat_im[self.roixi:self.roixf,self.roiyi:self.roiyf]))
            v = (100.*(kk+1))/len(self.roip)
            self.progress_norm.emit(v)
            kk = kk + 1
            
        self.normdone.emit()