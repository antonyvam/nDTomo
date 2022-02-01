# -*- coding: utf-8 -*-
"""
Misc tools for nDTomo

@author: Antony Vamvakeros
"""

import numpy as np
import matplotlib.pyplot as plt
import pkgutil, h5py
from pystackreg import StackReg
from scipy.interpolate import interp1d
from mayavi import mlab

def showplot(spectrum, fignum = 1):
    
    plt.figure(fignum);plt.clf()
    plt.plot(spectrum)
    plt.show()
    
def showim(im, fignum = 1, clim=None, cmap='jet'):
    
    plt.figure(fignum);plt.clf()
    plt.imshow(im, cmap = cmap)
    plt.colorbar()
    plt.axis('tight')
    if clim is None:
        plt.clim(np.min(im), np.max(im))
    else:
        plt.clim(clim)
    plt.show()
    
def showspectra(spectra, fignum = 1):
    
    plt.figure(fignum);plt.clf()
    for ii in range(len(spectra)):
        plt.plot(spectra[ii])
    plt.show()
    
def showvol(vol, vlim = None):
    
    '''
    Volume rendering using mayavi mlab
    '''    
    
    if vlim is None:
        
        vmin = 0
        vmax = np.max(vol)
    
    else:
        
        vmin, vmax = vlim
    
    mlab.pipeline.volume(mlab.pipeline.scalar_field(vol), vmin=vmin, vmax=vmax)
    
def closefigs():
    
    plt.close('all')
    
def plotfigs_imgs(imagelist, legendlist=None, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'jet'):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(legendlist)):

            legendlist.append('Component %d' %(ii+1))        
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(imagelist)):
            
            i = axes[ii].imshow(imagelist[ii], cmap=cmap)
            axes[ii].set_axis_off()
            axes[ii].set_title(legendlist[ii])
    
            if cl==True:
                fig.colorbar(i, ax=axes[ii])

    elif len(axes.shape)==2:
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                print(kk)
                
                if kk < len(imagelist):
            
                    i = axes[ii,jj].imshow(imagelist[kk], cmap=cmap)
                    axes[ii,jj].set_axis_off()
                    axes[ii,jj].set_title(legendlist[kk])
            
                    if cl==True:
                        fig.colorbar(i, ax=axes[ii,jj])        
                    
                    kk = kk + 1

def create_complist_imgs(components, xpix, ypix):     

    imagelist = []; legendlist = []
    for ii in range(components.shape[0]):
        im = components[ii,:]
        imagelist.append(np.reshape(im, (xpix, ypix)))
        legendlist.append('Component %d' %(ii+1))
        
    return(imagelist, legendlist)


    
def plotfigs_spectra(spectralist, legendlist= None, xaxis=None, rows=1, cols=5, figsize=(20,3)):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(legendlist)):

            legendlist.append('Component %d' %(ii+1))    
    
    if xaxis is None:
        
        xaxis = np.arange(0, len(spectralist[0]))
            
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(spectralist)):
            
            axes[ii].plot(xaxis, spectralist[ii])
            axes[ii].set_title(legendlist[ii])

    elif len(axes.shape)==2:
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                print(kk)
                
                if kk < len(spectralist):
            
                    axes[ii,jj].plot(xaxis, spectralist[kk])
                    axes[ii,jj].set_title(legendlist[kk])
   
                    kk = kk + 1
                    
def create_complist_spectra(components):     

    splist = []; legendlist = []
    for ii in range(components.shape[0]):
        splist.append(components[ii,:])
        legendlist.append('Component %d' %(ii+1))
        
    return(splist, legendlist)
    
def ndtomopath():
    
    '''
    Finds the absolute path of the nDTomo software
    '''
    
    package = pkgutil.get_loader('nDTomo')
    ndtomo_path = package.get_filename('nDTomo')
    ndtomo_path = ndtomo_path.split('__init__.py')[0]
            
    return(ndtomo_path)

def addpnoise1D(sp, ct):
    
    mi = np.min(sp)
    
    if mi < 0:
        
        sp = sp - sp + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        sp = sp + np.finfo(np.float32).eps
    
    return(np.random.poisson(sp * ct)/ ct)

def addpnoise2D(im, ct):
    
    mi = np.min(im)
    
    if mi < 0:
        
        im = im - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        im = im + np.finfo(np.float32).eps
    
    return(np.random.poisson(im * ct)/ ct)

def addpnoise3D(vol, ct):
    
    '''
    Adds poisspn noise to a stack of images, 3rd dimension is z/spectral
    '''
    
    mi = np.min(vol)
    
    if mi < 0:
        
        vol = vol - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        vol = vol + np.finfo(np.float32).eps
        
    
    for ii in range(vol.shape[2]):
        
        vol[:,:,ii] = np.random.poisson(vol[:,:,ii] * ct)/ ct
    
    
    return(vol)


def cirmask(im, npx=0):
    
    """
    
    Apply a circular mask to the image/volume
    
    """
    
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2));
    
    dim =  im.shape
    if len(dim)==2:
        im = np.where(r>np.floor(sz/2) - npx,0,im)
    elif len(dim)==3:
        for ii in range(0,dim[2]):
            im[:,:,ii] = np.where(r>np.floor(sz/2),0,im[:,:,ii])
    return(im)


def h5read(filename):
    df = {}
    with h5py.File(filename, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())
    
        # Get the data
        for key in a_group_key:
            
            df[key] = np.array(list(f[key][()]))
            
    return(df)


def regimage(ref, mov):

    '''
    Register an image using a reference image
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    reg = sr.register_transform(ref, mov)
    tmat = sr.register(ref, mov)
    reg = sr.transform(mov, tmat)
    
    plt.figure(1);plt.clf()
    plt.imshow(np.concatenate((ref, mov, reg), axis = 1), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()
    
    plt.figure(2);plt.clf()
    plt.imshow(np.concatenate((ref, mov, reg), axis = 0), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()
    
    plt.figure(3);plt.clf()
    plt.imshow(np.concatenate((mov - ref , reg - ref), axis = 1), cmap = 'jet')
    plt.clim(-0.3, 0.3)
    plt.colorbar()
    plt.show()

    return(reg, tmat)

def regvol(vol, tmat):
    
    '''
    Register a volume using a transformation matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    
    for ii in range(vol.shape[2]):
        
        vol[:,:,ii] = sr.transform(vol[:,:,ii], tmat)
        
        print(ii)

    return(vol)

def regimtmat(im, tmat):

    '''
    Register an image using a transformation matrix
    Uses rigid body transformation (i.e. translation/rotation only)
    '''
    
    sr = StackReg(StackReg.RIGID_BODY)
    
    reg = sr.transform(im, tmat)
    
    return(reg)

def maskvolume(vol, msk):
    
    '''
    Apply a mask to a 3D array
    It assumes that the spectral/heigh dimension is the 3rd dimension    
    '''
    voln = np.zeros_like(vol)
    
    for ii in range(vol.shape[2]):
        
        voln[:,:,ii] = vol[:,:,ii]*msk
        
    return(voln)

def interpvol(vol, xold, xnew, progress=False):
    
    '''
    Linear interpolation of a 3D matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension   
    '''
        
    voln = np.zeros((vol.shape[0], vol.shape[1], len(xnew)))
    
    for ii in range(voln.shape[0]):
        for jj in range(voln.shape[1]):
            
            f = interp1d(xold, vol[ii,jj,:], kind='linear', bounds_error=False, fill_value=0)
            voln[ii,jj,:] = f(xnew)    
    
        if progress == True:
            print('Interpolating line %s' %ii)
            
    return(voln)


def normvol(vol, progress=False):
    
    '''
    Normalise a 3D matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension   
    '''
        
    voln = np.zeros_like(vol)
    
    for ii in range(voln.shape[2]):

        voln[:,:,ii] = vol[:,:,ii]/np.max(vol[:,:,ii])   
    
        if progress == True:
            print('Interpolating line %s' %ii)
            
    return(voln)

def mask_thr(vol, roi, thr, fignum = 1):
    
    im = np.sum(vol[:,:,roi], axis = 2)
    im = im/np.max(im)
    msk = np.where(im<thr, 0, 1)

    plt.figure(fignum);plt.clf()
    plt.imshow(np.concatenate((im, msk), axis = 1), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()

    return(msk)


def tth2q(tth, E):

    """
	Convert 2theta to q
	"""  	
    
    h = 6.620700406E-34;c = 3E8
    wavel = 1E10*6.242E18*h*c/(E*1E3)
    q = np.pi*2/(wavel/(2*np.sin(np.deg2rad(0.5*tth))))

    return(q)

def q2tth(q, E):

    """
	Convert q to 2theta
	"""  	
    
    h = 6.620700406E-34;c = 3E8
    wavel = 1E10*6.242E18*h*c/(E*1E3)
    
    
    tth = np.rad2deg(2*np.arcsin(wavel/(4*np.pi/q)))

    return(tth)

def h5write_dataset(p, fn, data):

    '''
    Write a dataset to an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        data: dataset e.g. numpy array
    '''
    
    f = "%s%s.h5" %(p,fn)
    
    h5f = h5py.File(f, "w")
    
    h5f.create_dataset('data', data = data)
    
    h5f.close()


def h5read_dataset(p, fn, data = 'data'):

    '''
    Reads a dataset from an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        data: string corresponding to the h5 dataset
    '''
    
    f = "%s%s.h5" %(p,fn)
    
    with h5py.File(f, "r") as h5f:
    
        data = h5f[data][:]

    return(data)


















