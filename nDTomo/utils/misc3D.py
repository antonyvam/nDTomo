# -*- coding: utf-8 -*-
"""
Misc3D  tools for nDTomo

@author: Antony Vamvakeros
"""

from numpy import max
from mayavi import mlab

def showvol(vol, vlim = None):
    
    '''
    Volume rendering using mayavi mlab
    '''    
    
    if vlim is None:
        
        vmin = 0
        vmax = max(vol)
    
    else:
        
        vmin, vmax = vlim
    
    mlab.pipeline.volume(mlab.pipeline.scalar_field(vol), vmin=vmin, vmax=vmax)