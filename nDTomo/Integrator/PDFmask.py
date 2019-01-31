# -*- coding: utf-8 -*-
"""
Class to create a detector mask for pair distribution function data

@author: The methods of this class were written by G. Vaughan, class created by A. Vamvakeros
"""

from PyQt5.QtCore import QThread
from numpy import ogrid, logical_or, fliplr, tan, pi, sqrt
import pyFAI, fabio
from os import system

class CreatPDFMask(QThread):
    '''Class to create a detector mask for pair distribution function data'''
    
    def __init__(self,ponifile,mask):
        QThread.__init__(self)
        self.poniname = ponifile     
        self.mask = mask     
                
    def angle_mask(self,arr,tang,centre=None):
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
        if y0 > ny/2:
            y,x = ogrid[-x0:nx-x0,-y0:ny-y0]
        else:
            y,x = ogrid[-x0:nx-x0,y0-ny:y0]
            mask1 = y/x <= tang
            mask2 = x/y <= tang
            mask=logical_or(mask1,mask2)
        if y0 < ny/2:
            mask=fliplr(mask)
        
        self.amask = mask
    
    def radial_mask(self,arr,r,centre=None):
        # sum all the voxels with in a given radius
        # for now the centre is determined to be the centre of the array unless specified
        # print qarr.shape
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
            y,x = ogrid[-x0:nx-x0,-y0:ny-y0]
            mask = x*x + y*y >= r*r
        self.rmask = mask
        
    def block_mask(self,arr,tang,centre=None):
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
        if y0 > ny/2:
            y,x = ogrid[-x0:nx-x0,-y0:ny-y0]
            mask1 = x >= 0
            mask2 = y >= 0
        else:
            y,x = ogrid[-x0:nx-x0,-y0:ny-y0]
            mask1 = x <= 0
            mask2 = y >= 0
            mask = logical_or(mask1,mask2)
        self.bmask = mask
    
    def run(self):
        
        m=fabio.open(self.mask).data
        ai=pyFAI.load(self.poniname)
        alpha=float(75) #In degrees
        
        x0=ai.poni1/ai.pixel1
        y0=ai.poni2/ai.pixel2
        dy=1479-x0
        dx=1675-y0
        
        tang=tan((90.0-alpha)/2.0*pi/180.0)
        
        y=1479-dy-x0*tang
        y=dy+x0*tang
        x=1675-dx-y0*tang
        
        # so the points on the edge are
        # (x0,0), (0,y0)
        # which is shorter?
        
        d1x=(x0-x)
        d1y=(y0)
        d2y=(y0-y)
        d2x=(x0)
        
        l1=sqrt(d1x*d1x+d1y*d1y)
        l2=sqrt(d2x*d2x+d2y*d2y)
        
        # print x0,y0,x,y,tang
        
        # take off 5 pixels near the edge
        self.radial_mask(m,min(l1,l2)-5,(x0,y0))
        self.angle_mask(m,tang,(x0,y0))
        self.block_mask(m,tang,(x0,y0))
        	
        mask=logical_or.reduce((self.rmask,self.amask,self.bmask,m))
        #mask=np.logical_or.reduce((rmask,amask,m))
        outfile='PDFmask_%.1fdeg.cbf'%(alpha)
        outbuffer=fabio.cbfimage.cbfimage(data=mask)
        outbuffer.write(outfile)
        print 'Wrote mask to', outfile    
    
        perm = 'chmod 777 %s' %self.outfile
        system(perm) 