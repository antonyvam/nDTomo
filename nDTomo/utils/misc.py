# -*- coding: utf-8 -*-
"""
Misc tools for nDTomo

@author: Antony Vamvakeros
"""

import numpy as np
import matplotlib.pyplot as plt
import pkgutil
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure
from skimage.segmentation import flood, flood_fill
from scipy.optimize import minimize
   
def ndtomopath():
    
    '''
    Finds the absolute path of the nDTomo software
    '''
    
    package = pkgutil.get_loader('nDTomo')
    ndtomo_path = package.get_filename('nDTomo')
    ndtomo_path = ndtomo_path.split('nDTomo\__init__.py')[0]
            
    return(ndtomo_path)

def create_circle(npix_im=512, r0=128):
    
    """
    
    Create a circular mask for a squarred image
    
    """
    im = np.ones((npix_im, npix_im), dtype='float32')
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2))
    
    im = np.where(r>np.floor(sz/2) -(np.floor(sz/2) - r0)+1,0,1)
    im = np.where(r<np.floor(sz/2) -(np.floor(sz/2) - r0), 0, im)

    return(im)

def create_cirmask(npix_im=512, npx=0):
    
    """
    
    Create a circular mask for a squarred image
    
    """
    im = np.ones((npix_im, npix_im), dtype='float32')
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2))
    
    im = np.where(r>np.floor(sz/2) - npx,0,1)
    return(im)

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


def translate_xy(vol, npix):
    
    '''
    This is quite slow - has to be replaced with 3D interpolation
    '''
    
    xold = np.arange(vol.shape[1])
    xnew = np.arange(vol.shape[1]) + npix
    
    voln = np.zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[0])):
        
        im_tmp = vol[ii,:,:]
        
        for jj in range(im_tmp.shape[0]):
            
            f = interp1d(xold, im_tmp[jj,:], kind='linear', bounds_error=False, fill_value=0)
            voln[ii,jj,:] = f(xnew)    
            
    return(voln)

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

def mask_thr(vol, thr, roi=None, fignum = 1):
    
    dims = vol.shape
    if len(dims)==3:
        if roi is None:
            im = np.sum(vol, axis = 2)
        else:
            im = np.sum(vol[:,:,roi], axis = 2)
    elif len(dims)==2:            
        im = np.copy(vol)
    im = im/np.max(im)
    msk = np.where(im<thr, 0, 1)

    plt.figure(fignum);plt.clf()
    plt.imshow(np.concatenate((im, msk), axis = 1), cmap = 'jet')
    plt.colorbar()
    plt.axis('tight')
    plt.show()

    return(msk)

def calculate_center_of_mass(spectrum):
    """
    Calculate the center of mass of a spectrum with subpixel resolution.
    
    Arguments:
    spectrum -- A list or numpy array representing the spectrum.
    
    Returns:
    The center of mass of the spectrum with subpixel resolution.
    """
    spectrum = np.array(spectrum)
    
    # Calculate the total intensity of the spectrum
    total_intensity = np.sum(spectrum)
    
    # Calculate the weighted indices
    weighted_indices = np.arange(len(spectrum)) * spectrum
    
    # Calculate the center of mass with subpixel resolution
    center_of_mass = np.sum(weighted_indices) / total_intensity
    
    # Calculate the subpixel correction
    subpixel_correction = np.sum((np.arange(len(spectrum)) - center_of_mass) * spectrum) / (total_intensity * 2)
    
    # Calculate the final center of mass with subpixel resolution
    center_of_mass += subpixel_correction
    
    return center_of_mass

def compare_spectra(reference_spectrum, translated_spectrum, pixel_range, resolution):
    """
    Compare two spectra by translating the second spectrum with subpixel resolution using interpolation.
    
    Arguments:
    reference_spectrum -- A list or numpy array representing the reference spectrum.
    translated_spectrum -- A list or numpy array representing the spectrum to be translated.
    pixel_range -- The range of subpixel translation in pixels (e.g., (-0.5, 0.5)).
    resolution -- The resolution of subpixel translation (e.g., 0.1).
    
    Returns:
    The translated spectrum aligned with the reference spectrum.
    """
    reference_spectrum = np.array(reference_spectrum)
    translated_spectrum = np.array(translated_spectrum)
    
    # Calculate the subpixel translation range
    subpixel_range = np.arange(pixel_range[0], pixel_range[1] + resolution, resolution)
    
    best_shift = 0
    best_error = np.inf
    
    # Iterate through subpixel translations and find the best match
    for shift in subpixel_range:
        # Perform subpixel translation using interpolation
        translated_spectrum_shifted = interp1d(np.arange(len(translated_spectrum)), translated_spectrum, kind='cubic', fill_value=0.0, bounds_error=False)(np.arange(len(translated_spectrum)) + shift)
        
        # Calculate the error between the reference spectrum and translated spectrum
        error = np.sum(np.abs(reference_spectrum - translated_spectrum_shifted))
        
        # Update the best shift if the current error is lower
        if error < best_error:
            best_error = error
            best_shift = shift
    
    # Perform the final subpixel translation using interpolation
    translated_spectrum_aligned = interp1d(np.arange(len(translated_spectrum)), translated_spectrum, kind='cubic', fill_value=0.0, bounds_error=False)(np.arange(len(translated_spectrum)) + best_shift)
    
    return translated_spectrum_aligned

def sinocom_correction(sinograms):

    """
    Method correcting the sinograms for any motor jitter
    Sinogram can be a 2D or 3D matrix (stack of sinograms)
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """   
    
    di = sinograms.shape
    if len(di)>2:
        ss = np.sum(sinograms, axis = 2)
    else:
        ss = np.copy(sinograms)
            
    com = np.zeros((ss.shape[1],1))
    
    for ii in range(ss.shape[1]):
        
        com[ii,:] = calculate_center_of_mass(ss[:,ii])
        
    com = com - com[0]    
    
    sn = np.zeros_like(sinograms)
    
    xold = np.arange(sn.shape[0])


    if len(di)==2:
        
        for ii in tqdm(range(sn.shape[1])):
        
            xnew =  xold + com[ii,:]
                
            sn[:,ii] = np.interp(xnew, xold, sinograms[:,ii])    
        
    elif len(di)>2:
                
        for ll in tqdm(range(sinograms.shape[2])):
            
            for ii in range(sinograms.shape[1]):        
                        
                xnew =  xold + com[ii,:]
                    
                sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll])                    
        
        
    return(sn)


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


def even_idx(a):
    '''
    Returns all even elements from a matrix
    Answer to: https://stackoverflow.com/questions/41839634/numpy-array-select-all-even-elements-from-d-dimensional-array
    '''
    return a[np.ix_(*[range(0,i,2) for i in a.shape])]

def odd_idx(a):
    '''
    Returns all odd elements from a matrix
    '''
    return a[np.ix_(*[range(1,i,2) for i in a.shape])]

     
def rgb2gray(im):
    
    '''
    RBG image to grayscale using the luminosity method
    '''
    
    im = im[:,:0]*0.3 + im[:,:1]*0.59 + im[:,:2]*0.11
    
    return(im)
    

def crop_ctimage(im, plot=False):
    
    '''
    Crop a CT image using a square inside the reconstruction circle
    '''
    
    d = int(np.round((1 - np.cos(np.deg2rad(45)))*(im.shape[0]/2)))
    im = im[d:-d, d:-d]
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(im, cmap = 'gray')
        plt.colorbar()
        plt.show()  
        
    return(im)
    
def crop_image(im, thr=None, norm=False, plot=False, inds=None):

    '''
    Crop an image
    '''
    
    dims = im.shape
    
    if len(dims)==3:
        im = np.mean(im, axis = 2)
        imo = np.copy(im)
    else:
        imo = np.copy(im)
    
    if norm:
        im = im/np.max(im)
    
    if thr is not None:
        im[im<thr] = 0
    
    if inds is None:
        row_vector = np.sum(im, axis = 1)
        col_vector = np.sum(im, axis = 0)
    
        indr = [i for i, x in enumerate(row_vector) if x > 0]
        indc = [i for i, x in enumerate(col_vector) if x > 0]
        inds = [indr, indc]
    else:
        indr = inds[0]
        indc = inds[1]
        

    imc = imo[indr[0]:indr[-1], indc[0]:indc[-1]]

    if plot:
        
        plt.figure()
        plt.imshow(imo, cmap = 'gray')
        plt.colorbar()
        plt.show()        

        plt.figure()
        plt.imshow(imc, cmap = 'gray')
        plt.colorbar()
        plt.show() 
        
    return(imc, inds)    

    
def crop_volume(vol, thr=None, plot=False, dtype='float32', inds=None):

    '''
    Crops a data volume using the average image along the third dimension
    '''
    
    im = np.sum(vol, axis = 2)
    im = im/np.max(im)
    
    if thr is not None:
        im[im<thr] = 0
    
    if inds is None:
        row_vector = np.sum(im, axis = 1)
        col_vector = np.sum(im, axis = 0)
    
        indr = [i for i, x in enumerate(row_vector) if x > 0]
        indc = [i for i, x in enumerate(col_vector) if x > 0]
    else:
        indr = inds[0]
        indc = inds[1]

    lindr = len(np.arange(indr[0], indr[-1]+1))
    lindc = len(np.arange(indc[0], indc[-1]+1))
    
    volc = np.zeros((lindr, lindc, vol.shape[2]), dtype='float32')

    for ii in tqdm(range(vol.shape[2])):

        volc[:,:,ii] = vol[indr[0]:indr[-1]+1, indc[0]:indc[-1]+1, ii]
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(np.sum(volc, axis=2), cmap = 'gray')
        plt.colorbar()
        plt.show()  
        
    return(volc)

def crop_volume_getinds(vol, thr=None):

    '''
    Crops a data volume using the average image along the third dimension
    '''
    
    im = np.sum(vol, axis = 2)
    im = im/np.max(im)
    
    if thr is not None:
        im[im<thr] = 0
    
    row_vector = np.sum(im, axis = 1)
    col_vector = np.sum(im, axis = 0)

    indr = [i for i, x in enumerate(row_vector) if x > 0]
    indc = [i for i, x in enumerate(col_vector) if x > 0]
    return(indr, indc)
    
def crop_ctvolume(vol, plot=False):
    
    '''
    Crop a CT image using a square inside the reconstruction circle
    '''
    
    d = int(np.round((1 - np.cos(np.deg2rad(45)))*(vol.shape[0]/2)))
    vol = vol[d:-d, d:-d,:]
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(np.sum(vol, axis=2), cmap = 'gray')
        plt.colorbar()
        plt.show()  
        
    return(vol)


def fill_2d_binary(im, thr = None, dil_its = 2):
    
    '''
    Fill a 2D binary image
    '''
    
    im = im / np.max(im)
    if thr is not None:
        im[im<thr] = 0
    im[im>0] = 1
    
    struct = generate_binary_structure(2, 1).astype(im.dtype)
    
    for ii in range(dil_its):
        
        im = binary_dilation(im, structure=struct).astype(im.dtype)
    
    im[binary_fill_holes(im)] = 1
    
    return(im)

def image_segm(im, thr, norm = False, plot = False):
    
    '''
    Perform simple threshold based image segmentation
    '''
    
    imt = np.copy(im)

    if norm:
        imt = imt / np.max(imt)

    imt[imt<thr] = 0
    imt[imt>0] = 1
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(im, cmap = 'gray')
        plt.colorbar()
        plt.show()        

        plt.figure(2);plt.clf()
        plt.imshow(imt, cmap = 'gray')
        plt.colorbar()
        plt.show() 
        
    return(imt)


def make_matrix_even(mat):

    '''
    Makes a 1D, 2D or 3D array have dimension sizes equal to an even number
    '''
    
    dims = mat.shape

    if len(dims)==1:
        if np.mod(dims[0],2) != 0:
            mat = mat[1:]
        dims = mat.shape
        
    if len(dims)==2:
        if np.mod(dims[0],2) != 0:
            mat = mat[1:,:]
        if np.mod(dims[1],2) != 0:
            mat = mat[:,1:]
        dims = mat.shape
        
    if len(dims)==3: 
        if np.mod(dims[0],2) != 0:
            mat = mat[1:,:,:]
        if np.mod(dims[1],2) != 0:
            mat = mat[:,1:,:]
        if np.mod(dims[2],2) != 0:
            mat = mat[:,:,1:]
        dims = mat.shape        
       
    return(mat)
    
def pad_zeros_vol(vol1, vol2, dtype):

    # Make each dimension of both volumes an even number
    vol1 = make_matrix_even(vol1)
    vol2 = make_matrix_even(vol2)
    
    dims1 = vol1.shape
    dims2 = vol2.shape

    ofsx = int((dims1[0]-dims2[0])/2)
    ofsy = int((dims1[1]-dims2[1])/2)
    ofsz = int((dims1[2]-dims2[2])/2)

    if ofsx>0:
        zerosmat = np.zeros((ofsx, vol2.shape[1], vol2.shape[2]), dtype=dtype)
        vol2 = np.concatenate((zerosmat, vol2, zerosmat), axis = 0)
    elif ofsx<0:
        zerosmat = np.zeros((np.abs(ofsx), vol1.shape[1], vol1.shape[2]), dtype=dtype)
        vol1 = np.concatenate((zerosmat, vol1, zerosmat), axis = 0)

    if ofsy>0:
        zerosmat = np.zeros((vol2.shape[0], ofsx, vol2.shape[2]), dtype=dtype)
        vol2 = np.concatenate((zerosmat, vol2, zerosmat), axis = 1)
    elif ofsy<0:
        zerosmat = np.zeros((vol1.shape[0], np.abs(ofsy), vol1.shape[2]), dtype=dtype)
        vol1 = np.concatenate((zerosmat, vol1, zerosmat), axis = 1)

    if ofsz>0:
        zerosmat = np.zeros((vol2.shape[0], vol2.shape[1], ofsz), dtype=dtype)
        vol2 = np.concatenate((zerosmat, vol2, zerosmat), axis = 2)
    elif ofsz<0:
        zerosmat = np.zeros((vol1.shape[0], vol2.shape[2], np.abs(ofsz)), dtype=dtype)
        vol1 = np.concatenate((zerosmat, vol1, zerosmat), axis = 2)

    return(vol1, vol2)

def pad_zeros_svol(vol, dims, dtype):

    # Make each dimension of both volumes an even number
    vol = make_matrix_even(vol)
    
    dimsvol = vol.shape

    ofsx = int((dimsvol[0]-dims[0])/2)
    ofsy = int((dimsvol[1]-dims[1])/2)
    ofsz = int((dimsvol[2]-dims[2])/2)

    if ofsx<0:
        zerosmat = np.zeros((np.abs(ofsx), vol.shape[1], vol.shape[2]), dtype=dtype)
        vol = np.concatenate((zerosmat, vol, zerosmat), axis = 0)

    if ofsy<0:
        zerosmat = np.zeros((vol.shape[0], np.abs(ofsy), vol.shape[2]), dtype=dtype)
        vol = np.concatenate((zerosmat, vol, zerosmat), axis = 1)

    if ofsz<0:
        zerosmat = np.zeros((vol.shape[0], vol.shape[1], np.abs(ofsz)), dtype=dtype)
        vol = np.concatenate((zerosmat, vol, zerosmat), axis = 2)

    return(vol)


def pad_zeros(im1, im2, dtype):

    dims1 = im1.shape
    dims2 = im2.shape

    if np.mod(dims1[0],2) != 0:
        im1 = im1[1:,:]
        dims1 = im1.shape
    if np.mod(dims2[0],2) != 0:
        im2 = im2[1:,:]
        dims1 = im1.shape
    if np.mod(dims1[1],2) != 0:
        im1 = im1[:,1:]
        dims1 = im1.shape
    if np.mod(dims2[1],2) != 0:
        im2 = im2[:,1:]
        dims2 = im2.shape

    # Pad zeros rowise
    ofsr = int((dims1[0]-dims2[0])/2)
    if ofsr>0:
        
        zerosmat = np.zeros((ofsr, im2.shape[1]), dtype=dtype)
        im2 = np.concatenate((zerosmat, im2, zerosmat), axis = 0)
    
    elif ofsr<0:  

        zerosmat = np.zeros((np.abs(ofsr), im1.shape[1]), dtype=dtype)
        im1 = np.concatenate((zerosmat, im1, zerosmat), axis = 0)

    # Pad zeros colmnwise
    dims1 = im1.shape
    dims2 = im2.shape
    ofsc = int((dims1[1]-dims2[1])/2)
    if ofsc>0:
        
        zerosmat = np.zeros((im2.shape[0], np.abs(ofsc)), dtype=dtype)
        im2 = np.concatenate((zerosmat, im2, zerosmat), axis = 1)
    
    elif ofsc<0:  

        zerosmat = np.zeros((im1.shape[0], np.abs(ofsc)), dtype=dtype)
        im1 = np.concatenate((zerosmat, im1, zerosmat), axis = 1)

    return(im1, im2)


def nan_to_number(array, val=0):
    
    '''
    Replace NaNs with a number, default value is 0
    '''
    
    return(np.where(np.isnan(array), val, array))


def number_to_nan(array, val=0):
    
    '''
    Replace a number with NaN, default value is 0
    '''
    array[array==val] = np.nan
    return(array)
    


def find_first_neighbors_2D(mat, r, c):
    
    '''
    Takes a binary 2D matrix as input and the coordinates (row, column) and returns a list with the first neighbour elements that are non-zero
    '''
    
    dims = mat.shape
    
    el_list = [[r-1, c-1], [r-1, c], [r-1, c+1],
               [r, c-1], [r, c+1],
               [r+1, c-1], [r+1, c], [r+1, c+1]]

    ind_list = []
    
    for el in range(len(el_list)):
        
        x, y = el_list[el]
        
        if x>=0 and y>=0 and x<dims[0] and y<dims[1]:

            if mat[x,y] > 0:
                
                ind_list.append([x,y])

    return(ind_list)


def fit_shape_to_points(image, shape='circle'):
    """
    Fits a circle or ellipse to five user-selected points on an image and overlays it on the original image.

    Args:
        image (numpy.ndarray): 2D numpy array representing the image.
        shape (str): Shape to fit. Options: 'circle' (default) or 'ellipse'.

    Returns:
        None
    """

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title('Click five points on the image')
    plt.axis('image')

    # Wait for user input to select points
    plt.waitforbuttonpress()

    # Get the figure and axes
    fig = plt.gcf()
    ax = fig.gca()

    # Initialize a list to store the clicked points
    clicked_points = []

    def onclick(event):
        """
        Event handler for mouse click events.

        Args:
            event (matplotlib.backend_bases.MouseEvent): Mouse click event object.

        Returns:
            None
        """
        if len(clicked_points) < 5:
            # Append the clicked point to the list
            clicked_points.append((event.xdata, event.ydata))

            # Plot the clicked point
            ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
            plt.draw()

    # Connect the onclick event handler
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Wait for five points to be clicked
    while len(clicked_points) < 5:
        plt.waitforbuttonpress()

    # Extract the x and y coordinates from the clicked points
    x_coords, y_coords = zip(*clicked_points)

    # Fit a shape to the clicked points using least squares optimization
    if shape == 'circle':
        def shape_residuals(params):
            """
            Residual function for circle fitting.

            Args:
                params (numpy.ndarray): Circle parameters (x0, y0, r).

            Returns:
                float: Residual error.
            """
            x0, y0, r = params
            residuals = np.sqrt((x_coords - x0)**2 + (y_coords - y0)**2) - r
            return np.sum(residuals**2)
        
        initial_guess = np.mean(x_coords), np.mean(y_coords), np.mean([np.sqrt((x - np.mean(x_coords))**2 + (y - np.mean(y_coords))**2) for x, y in clicked_points])
        result = minimize(shape_residuals, initial_guess)
        
        # Extract the optimized circle parameters
        x0, y0, r = result.x

        # Plot the image with the fitted shape
        plt.imshow(image, cmap='gray')
        plt.title('Image with Fitted Circle')
        plt.axis('image')

        # Plot the circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)
        plt.plot(x, y, 'r-', label='Fitted Circle')
        plt.legend()

    elif shape == 'ellipse':
        def shape_residuals(params):
            """
            Residual function for ellipse fitting.

            Args:
                params (numpy.ndarray): Ellipse parameters (x0, y0, a, b).

            Returns:
                float: Residual error.
            """
            x0, y0, a, b = params
            residuals = (((x_coords - x0) / a)**2 + ((y_coords - y0) / b)**2) - 1
            return np.sum(residuals**2)
        
        initial_guess = np.mean(x_coords), np.mean(y_coords), np.std(x_coords), np.std(y_coords)
        result = minimize(shape_residuals, initial_guess)
        
        # Extract the optimized ellipse parameters
        x0, y0, a, b = result.x

        # Plot the image with the fitted shape
        plt.imshow(image, cmap='gray')
        plt.title('Image with Fitted Ellipse')
        plt.axis('image')

        # Plot the ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        x = x0 + a * np.cos(theta)
        y = y0 + b * np.sin(theta)
        plt.plot(x, y, 'r-', label='Fitted Ellipse')
        plt.legend()

    # Show the plot
    plt.show()