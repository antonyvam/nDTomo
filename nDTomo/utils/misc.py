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
from matplotlib import cm
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.interpolate import griddata

def nDTomo_colormap():
    
    '''
    Custom colormap: It is the jet colormap but 0 correponds to black
    '''
    
    jet = cm.get_cmap('jet', 256)
    newcolors = jet(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[0, :] = black
    newcmp = ListedColormap(newcolors)
    return(newcmp)

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
       
def closefigs():
    
    plt.close('all')
    
def plotfigs_imgs(imagelist, legendlist=None, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'jet'):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(imagelist)):

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
    ndtomo_path = ndtomo_path.split('nDTomo\__init__.py')[0]
            
    return(ndtomo_path)

def addpnoise1D(sp, ct):
    
    '''
    Adds poisson noise to a spectrum
    '''    
    mi = np.min(sp)
    
    if mi < 0:
        
        sp = sp - sp + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        sp = sp + np.finfo(np.float32).eps
    
    return(np.random.poisson(sp * ct)/ ct)

def addpnoise2D(im, ct):
    
    '''
    Adds poisson noise to an image
    '''
    
    mi = np.min(im)
    
    if mi < 0:
        
        im = im - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        im = im + np.finfo(np.float32).eps
    
    return(np.random.poisson(im * ct)/ ct)

def addpnoise3D(vol, ct):
    
    '''
    Adds poisson noise to a stack of images, 3rd dimension is z/spectral
    '''
    
    mi = np.min(vol)
    
    if mi < 0:
        
        vol = vol - mi + np.finfo(np.float32).eps
        
    elif mi == 0:
        
        vol = vol + np.finfo(np.float32).eps
        
    
    for ii in tqdm(range(vol.shape[2])):
        
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
        for ii in tqdm(range(0,dim[2])):
            im[:,:,ii] = np.where(r>np.floor(sz/2) - npx,0,im[:,:,ii])
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
    
    for ii in tqdm(range(vol.shape[2])):
        
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
    
    for ii in tqdm(range(vol.shape[2])):
        
        voln[:,:,ii] = vol[:,:,ii]*msk
        
    return(voln)

def interpvol(vol, xold, xnew):
    
    '''
    Linear interpolation of a 3D matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension   
    '''
        
    voln = np.zeros((vol.shape[0], vol.shape[1], len(xnew)))
    
    for ii in tqdm(range(voln.shape[0])):
        for jj in range(voln.shape[1]):
            
            f = interp1d(xold, vol[ii,jj,:], kind='linear', bounds_error=False, fill_value=0)
            voln[ii,jj,:] = f(xnew)    
    
    return(voln)


def normvol(vol):
    
    '''
    Normalise a 3D matrix
    It assumes that the spectral/heigh dimension is the 3rd dimension   
    '''
        
    voln = np.zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[2])):

        voln[:,:,ii] = vol[:,:,ii]/np.max(vol[:,:,ii])   
                
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

def matsum(mat, axes = [0,1], method = 'sum'):

    '''
    Dimensionality redunction of a multidimensional array
    Inputs:
        mat: the nD array
        axes: a list containing the axes along which the operation will take place
        method: the type of operation, options are 'sum' and 'mean'
    '''
    
    naxes = len(axes)
    squeezed = np.copy(mat)
    
    for ii in range(naxes):
        
        if method == 'sum':
        
            squeezed = np.sum(squeezed, axis = axes[ii])
            
        elif method == 'mean':
            
            squeezed = np.mean(squeezed, axis = axes[ii])            
    
    return(squeezed)

def KeVtoAng(E):
    
    """
	Convert photon energy in KeV to Angstrom
	"""  
    
    h = 6.620700406E-34;c = 3E8
    return(1E10*6.242E18*h*c/(E*1E3))

def AngtoKeV(wavel):
    
    """
	Convert photon energy in KeV to Angstrom
	"""  
    
    h = 6.620700406E-34;c = 3E8
    return(1E10*6.242E18*h*c/(wavel*1E3))

def tth2q(tth, E):

    """
	Convert 2theta to q
    E is energy in KeV
	"""  	
    
    wavel = KeVtoAng(E)
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

def h5write_data(p, fn, dlist, data):

    '''
    Write a dataset to an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        dlist: list containing the names for the various datasets
        data:list containing the various dataset e.g. list of numpy arrays
    '''
        
    f = "%s%s.h5" %(p,fn)
    
    h5f = h5py.File(f, "w")
    
    for ii in range(len(dlist)):
    
        h5f.create_dataset(dlist[ii], data = data[ii])
    
    h5f.close()



def h5read_data(p, fn, dlist):

    '''
    Reads a dataset from an h5 file.
    Inputs:
        p: path to dataset, e.g. 'C:\\path_to_data\\'
        fn: filename, e.g. 'file'
        dlist: list containing the names for the various datasets
    Outputs:
        data: dictionary containing the various datasets
    '''
    
    f = "%s%s.h5" %(p,fn)
    
    with h5py.File(f, "r") as h5f:
    
        data = {}
        
        for ii in range(len(dlist)):

            data[dlist[ii]] = h5f[dlist[ii]][:]

    return(data)

def cart2pol(x, y):
    
    '''
    Convert cartesian (x,y) coordinates to polar coordinates (rho, phi)
    '''
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    return(phi, rho)

def pol2cart(phi, rho):

    '''
    Convert polar (rho, phi) coordinates to cartesian coordinates (x,y)
    '''

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return(x, y)


def cart2polim(im, thpix=1024, rpix=1024, ofs=0):
    
    '''
    Converts an image from cartestian to polar coordinates
    Inputs:
        im: 2D array corresponding to the image
        thpix: number of bins for the azimuthal range, default=1024
        rpix: number of bins for the r distance range, default=1024
        ofs: angular offset, default=0
    '''
    
    x = np.arange(0, im.shape[0]) - im.shape[0]/2
    y = np.arange(0, im.shape[1]) - im.shape[1]/2
    xo, yo = np.meshgrid(x,y)
    xo = np.reshape(xo, (xo.shape[0]*xo.shape[1]))
    yo = np.reshape(yo, (yo.shape[0]*yo.shape[1]))
    imo = np.reshape(im, (im.shape[0]*im.shape[1]))
    
    
    xi = np.linspace((-1+ofs)*np.pi, (1+ofs)*np.pi, thpix)
    yi = np.linspace(0, int(np.floor(im.shape[0]/2)), rpix)
    xp, yp = np.meshgrid(xi,yi)
    xx, yy = pol2cart(xp,yp)
    
    imp = griddata((xo, yo), imo, (xx, yy), method='nearest')

    return(imp)

def pol2cartim(imp, im_size=None, thpix=1024, rpix=1024, ofs=0):
    
    '''
    Converts an image from polar to cartestian coordinates
    Inputs:
        imp: 2D array corresponding to the polar transformed image
        im_size: list containing the two dimensions of the image with cartesian coordinates
        thpix: number of bins for the azimuthal range, default=1024
        rpix: number of bins for the r distance range, default=1024
        ofs: angular offset, default=0
    '''
    if im_size is None:
        im_size = [imp.shape[0], imp.shape[1]]

    r = np.linspace((-1+ofs)*np.pi, (1+ofs)*np.pi, thpix)
    t = np.linspace(0, int(np.floor(im_size[0]/2)), rpix)
    rr, tt = np.meshgrid(r,t)
    ro = np.reshape(rr, (rr.shape[0]*rr.shape[1]))
    to = np.reshape(tt, (tt.shape[0]*tt.shape[1]))
    imo = np.reshape(imp, (imp.shape[0]*imp.shape[1]))
    
    x = np.arange(0, im_size[0]) - im_size[0]/2
    y = np.arange(0, im_size[1]) - im_size[1]/2    
    xc, yc = np.meshgrid(x,y)
    xx, yy = cart2pol(xc,yc)
    
    imn = griddata((ro, to), imo, (xx, yy), method='nearest')

    return(imn)
