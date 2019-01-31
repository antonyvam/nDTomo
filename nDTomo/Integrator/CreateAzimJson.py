# -*- coding: utf-8 -*-
"""
Class to create .azimint.json file for 

@author: A. Vamvakeros
"""

import json
from os import system
from PyQt5.QtCore import QThread

class CreatAzimint(QThread):
    
    def __init__(self,ponifile,mask,npt_rad):
        
        QThread.__init__(self)
        self.poniname = ponifile     
        self.mask = mask   
        self.npt_rad = npt_rad
        self.jsonname = '.azimint.json'
        
    def run(self):

        data = {}

        data["poni"] = self.poniname
        data['poni_file'] = self.poniname
        data['do_SA'] = 0
        data['method'] = "ocl_csr_nosplit_gpu"
        data['npt'] = self.npt_rad 
        data["chi_discontinuity_at_0"] = False
        data["do_mask"] = True 
        data["do_dark"] = False 
        data["do_azimuthal_range"] = False 
        data["do_flat"] = False 
        data["do_2D"] = False
        data["splineFile"] = "" 
        data["do_OpenCL"] = False
        data["pixel1"] = 0.000172 
        data["pixel2"] = 0.000172 
        data["polarization_factor"] = 0.97 
        data["do_solid_angle"] = False 
        data["do_radial_range"] = False 
        data["do_poisson"] = False 
        data["flat_field"] = "" 
        data["nbpt_rad"] = self.npt_rad 
        data["dark_current"] = "" 
        data["do_polarization"] = True 
        data["detector"] = "detector"
        data["unit"] = "2th_deg"
        data["do_dummy"] = False
        data["mask_file"] = self.mask        
        
        with open(self.poniname, 'r') as poni:
            for line in poni:
                if 'Wavelength' in line:
                    args=line.split()
                    Wavelength = float(args[1])
                if 'Distance' in line:
                    args=line.split()
                    Distance = float(args[1])
                if 'Poni1' in line:
                    args=line.split()
                    Poni1 = float(args[1])
                if 'Poni2' in line:
                    args=line.split()
                    Poni2 = float(args[1])
                if 'Rot1' in line:
                    args=line.split()
                    Rot1 = float(args[1])
                if 'Rot2' in line:
                    args=line.split()
                    Rot2 = float(args[1])
                if 'Rot3' in line:
                    args=line.split()
                    Rot3 = float(args[1])
                    
            print Wavelength
            print Distance
            print Poni1
            print Poni2
            print Rot1
            print Rot2
            print Rot3
            
        data["wavelength"] = Wavelength
        data["rot1"] = Rot1
        data["rot2"] = Rot2
        data["rot3"] = Rot3
        data["poni1"] = Poni1
        data["poni2"] = Poni2
        data["dist"] = Distance
            
        
        with open(self.jsonname, 'w') as f:
            json.dump(data, f)
        
        perm = 'chmod 777 %s' %self.jsonname
        system(perm)    