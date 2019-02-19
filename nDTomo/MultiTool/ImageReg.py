# -*- coding: utf-8 -*-
"""

Image registration class for the MultiTool

@author: A. Vamvakeros

"""

from PyQt5.QtCore import pyqtSignal, QThread
from skimage.transform import iradon, radon
from numpy import concatenate, log, flipud, zeros, sqrt, sum, arange, min, max, floor, where, mean, array, exp, inf, ceil, interp, std, argmin, transpose, tile, swapaxes, round

class AlignImages(QThread):
    
    '''
    
    The AlignImages class allows for aligning two images. The image registration is performed by translation and rotation.
    
    :theta: tomographic angles
            
    :iman: absorption-contrast CT image
        
    :bpx: chemical-contrast CT image
    
    '''
    
    aligndone = pyqtSignal()
    progress_al = pyqtSignal(int)
    
    def __init__(self,theta,iman,bpx):
        QThread.__init__(self)
        self.theta = theta 
        self.iman = iman
        self.bpx = bpx
        
    def run(self):
                   
        """
        
        Initialise the image aligning process
        
        """  
        
        #Rotate the micro-CT image
        
        dia = []; ang = []
        kk = 0
        fpa = radon(self.iman, theta=self.theta, circle=True)
        roi_an = arange(-11,11,0.1)
        
        for ii in roi_an:
            
            thetan = self.theta + ii
            
            imn = iradon(fpa, theta=thetan, circle=True)
            imn = where(imn<0,0,imn)
            imn = imn/max(imn)        
            
            
            dia.append(mean((abs(imn-self.bpx))))
            ang.append(ii)
            kk = kk + 1
            
            v = (100.*(kk+1))/(len(roi_an))
            self.progress_al.emit(v)
            
        m = argmin(dia)
        print(ang[m])
        
        thetan = self.theta + ang[m]
        self.iman = iradon(fpa, theta=thetan, circle=True)
        self.iman = where(self.iman<0,0,self.iman)
        self.iman = self.iman/max(self.iman)   
        
        self.aligndone.emit()  