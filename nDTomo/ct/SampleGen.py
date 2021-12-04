# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:00:58 2020

@author: Antony
"""

from skimage.draw import random_shapes
import numpy as np
import astra, time, h5py

class random_sample(object):
    
    def __init__(self, *args, **kwargs):
        
        super(random_sample, self).__init__(*args, **kwargs)

        self.sz = 50
        self.misz = 2
        self.masz = 15
        self.mxshapes = 50
        
    def create_image(self, sz=50, misz=2, masz=15, mxshapes=50):
        
        
        self.im, _ = random_shapes((sz, sz), min_shapes=1, max_shapes=mxshapes, multichannel=False,
                         min_size=misz, max_size=masz, allow_overlap=True)
        self.im = np.where(self.im==255, 0, self.im)
        
        self.im = self.cirmask(5)
        if np.max(self.im)>0:
            self.im = self.im/np.max(self.im)
        
        return(self.im)
        
    def cirmask(self, npx=0):
        
        """
        
        Apply a circular mask to the image
        
        """
        
        sz = np.floor(self.im.shape[0])
        x = np.arange(0,sz)
        x = np.tile(x,(int(sz),1))
        y = np.swapaxes(x,0,1)
        
        xc = np.round(sz/2)
        yc = np.round(sz/2)
        
        r = np.sqrt(((x-xc)**2 + (y-yc)**2));
        
        dim =  self.im.shape
        if len(dim)==2:
            self.im = np.where(r>np.floor(sz/2) - npx,0,self.im)
        elif len(dim)==3:
            for ii in range(0,dim[2]):
                self.im[:,:,ii] = np.where(r>np.floor(sz/2),0,self.im[:,:,ii])
        return(self.im)

    def create_imagestack(self, nim=10):
        
        self.nim = nim
        self.vol = np.zeros((self.sz, self.sz, self.nim))
        
        for ii in range(self.nim):
            
            self.vol[:,:,ii] = self.create_image(sz=self.sz, misz=self.misz, masz=self.masz, mxshapes=self.mxshapes)
            
            if np.mod(ii, 1000) == 0:
                print('Image  %d out of %d' %(ii, self.nim))   
                
    def create_sino_geo(self):
        
        # Create a basic square volume geometry
        self.vol_geom = astra.create_vol_geom(self.im.shape[0], self.im.shape[0])
        # Create a parallel beam geometry with 180 angles between 0 and pi, and
        # im.shape[0] detector pixels of width 1.
        # For more details on available geometries, see the online help of the
        # function astra_create_proj_geom .
        self.theta = np.linspace(0,np.pi,self.im.shape[0])
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, self.im.shape[0], self.theta, False)
        # Create a sinogram using the GPU.
        # Note that the first time the GPU is accessed, there may be a delay
        # of up to 10 seconds for initialization.
        self.proj_id = astra.create_projector('cuda',self.proj_geom,self.vol_geom)        
    
    def create_sino(self):

        start=time.time()
        
        self.sinogram_id, self.sinogram = astra.create_sino(self.im, self.proj_id)
        
        print((time.time()-start))
        
        # self.sinogram = np.random.poisson(self.sinogram)
        
        self.sinogram = self.sinogram/np.max(self.sinogram)
        
        return(self.sinogram)
                
    def create_sinostack(self):
    
        self.sinograms = np.zeros((self.sz, len(self.theta), self.nim))
        
        for ii in range(self.nim):
            
             self.sinogram_id, self.sinograms[:,:,ii] = astra.create_sino(self.vol[:,:,ii], self.proj_id)
             
             if np.mod(ii, 1000) == 0:
                 print('Sinogram %d out of %d' %(ii, self.nim))
    
    def export_sinostack(self, savepath, dataset):
        
        """
        
		Export the sinogram stack as a single .hdf5 file.
        
		"""  
        
        fn = "%s\\%s_sinostack.h5" % (savepath, dataset)

        h5f = h5py.File(fn, "w")
        
        h5f.create_dataset('data', data=self.sinograms)
    
    def export_imagestack(self, savepath, dataset):
        
        """
        
		Export the image stack as a single .hdf5 file.
        
		"""  
        
        fn = "%s\\%s_imagestack.h5" % (savepath, dataset)

        h5f = h5py.File(fn, "w")
        
        h5f.create_dataset('data', data=self.vol)    
    
    def astraclean(self):
        
        astra.data2d.delete(self.sinogram_id)
        astra.projector.delete(self.proj_id)
    