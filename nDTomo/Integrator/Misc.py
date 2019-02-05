# -*- coding: utf-8 -*-
"""
Miscellaneous classes

@author: Antony
"""

from numpy import transpose

class SliceTransformations():
    
    def __init__(self):
        pass
    def rowFlip(self):
        self.data[1::2,:,:] = self.data[1::2,::-1,:]
        
    def colFlip(self):
        self.data[:,1::2,:] = self.data[::-1,1::2,:]

    def transpose2D(self):
	self.data = transpose(self.data,(1,0))

    def transpose3D(self):
	self.data = transpose(self.data,(1,0,2))
