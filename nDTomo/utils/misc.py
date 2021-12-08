# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:00:09 2021

@author: Antony
"""

import pkgutil

def ndtomopath():
    
    '''
    Finds the absolute path of the nDTomo software
    '''
    
    package = pkgutil.get_loader('nDTomo')
    ndtomo_path = package.path
    ndtomo_path = ndtomo_path.split('__init__.py')[0]
            
    return(ndtomo_path)
