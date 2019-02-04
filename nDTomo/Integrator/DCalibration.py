# -*- coding: utf-8 -*-
"""
Calibration class calls pyFAI-calib to create the detector calibration
CreatMask class creates a detector mask

@author: A. Vamvakeros
"""

from os import system
import fabio
from PyQt5.QtCore import QThread
from numpy import where, array, ones
from scipy.ndimage import binary_dilation

class Calibration(QThread):
    
    def __init__(self,calibrant,energy):
		"""
		:calibrant: experimental file (i.e. calibrant diffraction pattern)
		:energy: X-ray energy in keV
		"""
        QThread.__init__(self)
        self.calibrant = calibrant
        self.E = energy
        
    def run(self):
		"""
		Run pyFAI-calib (detector calibration)
		"""
        cmd = 'pyFAI-calib -e %.f -c CeO2 -D Pilatus2M_CdTe %s' %(self.E,self.calibrant)
        system(cmd)
    
       
class CreatMask(QThread):
    
    def __init__(self,calibrant):
		"""
		:calibrant: experimental file (i.e. calibrant diffraction pattern)
		"""	
        QThread.__init__(self)
        self.calibrant = calibrant
        
        mask_file = '/data/id15/inhouse2/inhouse03/gmask24.10.17.edf' # Static mask for ID15A only

        fm=fabio.open(mask_file)
        fd=fabio.open(self.calibrant)
        dmask=binary_dilation(fm.data,structure=ones((1,1))).astype(int)
        outdata=where(array(dmask)==1,-1,array(fd.data))
        outbuffer=fabio.edfimage.edfimage(data=outdata,header=fd.header)
        self.newcalibrant=self.calibrant[0:-4]+'_masked.edf'
        outbuffer.write(self.newcalibrant)

        perm = 'chmod 777 %s' %self.newcalibrant
        system(perm) 
        
    def run(self):
		"""
		Run pyFAI-drawmask (detector calibration)
		"""	
        cmd = 'pyFAI-drawmask %s' %(self.newcalibrant)
        system(cmd)