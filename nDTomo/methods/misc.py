# -*- coding: utf-8 -*-
"""
nDTomo Utility Tools
====================

This module contains a broad collection of general-purpose functions and tools
used across the nDTomo package. These include image and volume processing
utilities, geometric transformations, interpolation routines, masking,
thresholding, and auxiliary tools for rebinning, visualization, and user-guided
input.

Key functionalities:
- Circular and binary masking for 2D/3D data
- Coordinate transformations (Cartesian <-> Polar)
- Cropping, normalization, and padding of images and volumes
- Interpolation in 1D, 2D, and 3D
- Image segmentation, thresholding, and morphological filling
- Matrix size harmonization (even sizing, padding)
- Spectral registration and rebinding
- Interactive shape fitting (circle/ellipse)
- Miscellaneous utility functions (e.g., RGB to grayscale)

These tools serve as foundational operations for pre-processing, simulation,
and reconstruction workflows in X-ray chemical tomography.

@author: Antony Vamvakeros
"""

import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes, binary_dilation, generate_binary_structure
from scipy.optimize import minimize
   
def ndtomopath():
    """
    Returns the absolute path to the installed nDTomo package directory.

    This is useful for locating internal resources such as 'examples/' that now live inside the package.

    Returns:
        str: Absolute path to the nDTomo package.
    """
    spec = importlib.util.find_spec('nDTomo')
    if spec is None or spec.origin is None:
        raise ImportError("nDTomo package not found.")

    # Path to the nDTomo package folder
    package_dir = os.path.dirname(spec.origin)
    
    return package_dir

def create_circle(npix_im=512, r0=128):
    
    """
    Create a circular mask for a squared image.

    Args:
        npix_im (int): Size of the squared image (default: 512).
        r0 (int): Radius of the circular mask (default: 128).

    Returns:
        numpy.ndarray: A binary circular mask with the specified radius.

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
    Create a circular mask for a squared image.

    Args:
        npix_im (int): Size of the squared image (default: 512).
        npx (int): Number of pixels to exclude from the circular mask (default: 0).

    Returns:
        numpy.ndarray: A binary circular mask with excluded pixels.

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
    Apply a circular mask to the image/volume.

    Args:
        im (numpy.ndarray): Input image or volume.
        npx (int): Number of pixels to exclude from the circular mask (default: 0).

    Returns:
        numpy.ndarray: Image or volume with a circular mask applied.
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
    
    """
    Translate the volume along the x-axis by a specified number of pixels.

    Args:
        vol (numpy.ndarray): Input volume.
        npix (int): Number of pixels to translate the volume along the x-axis.

    Returns:
        numpy.ndarray: Translated volume.

    """
    
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
    
    """
    Apply a mask to a 3D array.

    Args:
        vol (numpy.ndarray): Input 3D array.
        msk (numpy.ndarray): Mask array.

    Returns:
        numpy.ndarray: Resulting masked 3D array.

    """
    voln = np.zeros_like(vol)
    
    for ii in tqdm(range(vol.shape[2])):
        
        voln[:,:,ii] = vol[:,:,ii]*msk
        
    return(voln)

def interpvol(vol, xold, xnew):
    
    """
    Perform linear interpolation on a 3D matrix along the spectral/height dimension.

    Args:
        vol (numpy.ndarray): Input 3D matrix.
        xold (numpy.ndarray): Original x-coordinates along the spectral/height dimension.
        xnew (numpy.ndarray): New x-coordinates for interpolation.

    Returns:
        numpy.ndarray: Interpolated 3D matrix.

    """
        
    voln = np.zeros((vol.shape[0], vol.shape[1], len(xnew)))
    
    for ii in tqdm(range(voln.shape[0])):
        for jj in range(voln.shape[1]):
            
            f = interp1d(xold, vol[ii,jj,:], kind='linear', bounds_error=False, fill_value=0)
            voln[ii,jj,:] = f(xnew)    
    
    return(voln)


def normvol(vol):
    
    """
    Normalize a 3D matrix along the spectral/height dimension.

    Args:
        vol (numpy.ndarray): Input 3D matrix.

    Returns:
        numpy.ndarray: Normalized 3D matrix.

    """
        
    voln = np.zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[2])):

        voln[:,:,ii] = vol[:,:,ii]/np.max(vol[:,:,ii])   
                
    return(voln)

def mask_thr(vol, thr, roi=None, fignum = 1):
    
    """
    Apply a threshold-based mask to a 2D or 3D volume.

    Args:
        vol (numpy.ndarray): Input 2D or 3D volume.
        thr (float): Threshold value.
        roi (numpy.ndarray, optional): Region of interest (default: None).
        fignum (int, optional): Figure number for displaying the plot (default: 1).

    Returns:
        numpy.ndarray: Binary mask based on the threshold.

    """
    
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
    
    Args:
        spectrum (list or numpy.ndarray): A list or numpy array representing the spectrum.
    
    Returns:
        float: The center of mass of the spectrum with subpixel resolution.
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
    
    Args:
        reference_spectrum (list or numpy.ndarray): A list or numpy array representing the reference spectrum.
        translated_spectrum (list or numpy.ndarray): A list or numpy array representing the spectrum to be translated.
        pixel_range (tuple): The range of subpixel translation in pixels (e.g., (-0.5, 0.5)).
        resolution (float): The resolution of subpixel translation (e.g., 0.1).
    
    Returns:
        numpy.ndarray: The translated spectrum aligned with the reference spectrum.
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


def cart2pol(x, y):
    
    """
    Convert Cartesian (x,y) coordinates to polar coordinates (rho, phi).
    
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
    
    Returns:
        tuple: Polar coordinates (phi, rho).
    """
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    return(phi, rho)

def pol2cart(phi, rho):

    """
    Convert polar (rho, phi) coordinates to Cartesian coordinates (x, y).
    
    Args:
        phi (float): Angle in radians.
        rho (float): Distance from the origin.
    
    Returns:
        tuple: Cartesian coordinates (x, y).
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return(x, y)


def cart2polim(im, thpix=1024, rpix=1024, ofs=0):
    
    """
    Convert an image from Cartesian to polar coordinates.
    
    Args:
        im (numpy.ndarray): 2D array corresponding to the image.
        thpix (int): Number of bins for the azimuthal range (default: 1024).
        rpix (int): Number of bins for the r distance range (default: 1024).
        ofs (float): Angular offset (default: 0).
    
    Returns:
        numpy.ndarray: Image in polar coordinates.
    """
    
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
    
    """
    Convert an image from polar to Cartesian coordinates.
    
    Args:
        imp (numpy.ndarray): 2D array corresponding to the polar-transformed image.
        im_size (list): List containing the two dimensions of the image with Cartesian coordinates (default: None).
        thpix (int): Number of bins for the azimuthal range (default: 1024).
        rpix (int): Number of bins for the r distance range (default: 1024).
        ofs (float): Angular offset (default: 0).
    
    Returns:
        numpy.ndarray: Image in Cartesian coordinates.
    """
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
    """
    Return all even elements from a matrix.
    Args:
        a (numpy.ndarray): Input matrix.
    Returns:
        numpy.ndarray: Array containing only the even elements.
    """
    return a[np.ix_(*[range(0,i,2) for i in a.shape])]

def odd_idx(a):
    """
    Return all odd elements from a matrix.
    Args:
        a (numpy.ndarray): Input matrix.
    Returns:
        numpy.ndarray: Array containing only the odd elements.
    """
    return a[np.ix_(*[range(1,i,2) for i in a.shape])]

     
def rgb2gray(im):
    
    """
    Convert an RGB image to grayscale using the luminosity method.
    Args:
        im (numpy.ndarray): RGB image.
    Returns:
        numpy.ndarray: Grayscale image.
    """
    
    im = im[:,:0]*0.3 + im[:,:1]*0.59 + im[:,:2]*0.11
    
    return(im)
    

def crop_ctimage(im, plot=False):
    
    """
    Crop a CT image using a square inside the reconstruction circle.
    Args:
        im (numpy.ndarray): CT image.
        plot (bool): Flag to display a plot of the cropped image (default: False).
    
    Returns:
        numpy.ndarray: Cropped CT image.
    """
    
    d = int(np.round((1 - np.cos(np.deg2rad(45)))*(im.shape[0]/2)))
    im = im[d:-d, d:-d]
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(im, cmap = 'gray')
        plt.colorbar()
        plt.show()  
        
    return(im)
    
def crop_image(im, thr=None, norm=False, plot=False, inds=None):

    """
    Crop an image.
    
    Args:
        im (numpy.ndarray): Input image.
        thr (float): Threshold value to apply for cropping (default: None).
        norm (bool): Flag to normalize the image (default: False).
        plot (bool): Flag to display plots of the original and cropped images (default: False).
        inds (list): List of indices for cropping (default: None).
    
    Returns:
        numpy.ndarray: Cropped image.
        list: List of indices used for cropping.
    """
    
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

    """
    Crop a data volume using the average image along the third dimension.
    
    Args:
        vol (numpy.ndarray): Input data volume.
        thr (float): Threshold value to apply for cropping (default: None).
        plot (bool): Flag to display a plot of the cropped volume (default: False).
        dtype (str): Data type of the cropped volume (default: 'float32').
        inds (list): List of indices for cropping (default: None).
    
    Returns:
        numpy.ndarray: Cropped data volume.
    """
    
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

    """
    Crop a data volume using the average image along the third dimension and return the indices.
    
    Args:
        vol (numpy.ndarray): Input data volume.
        thr (float): Threshold value to apply for cropping (default: None).
    
    Returns:
        tuple: Tuple of lists containing the row and column indices used for cropping.
    """
    
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
    
    """
    Crop a CT volume using a square inside the reconstruction circle.
    
    Args:
        vol (numpy.ndarray): Input CT volume.
        plot (bool): Flag to display a plot of the cropped volume (default: False).
    
    Returns:
        numpy.ndarray: Cropped CT volume.
    """
    
    d = int(np.round((1 - np.cos(np.deg2rad(45)))*(vol.shape[0]/2)))
    vol = vol[d:-d, d:-d,:]
    
    if plot:
        
        plt.figure(1);plt.clf()
        plt.imshow(np.sum(vol, axis=2), cmap = 'gray')
        plt.colorbar()
        plt.show()  
        
    return(vol)


def fill_2d_binary(im, thr = None, dil_its = 2):
    
    """
    Fill a 2D binary image.
    
    Args:
        im (numpy.ndarray): Input binary image.
        thr (float): Threshold value to apply for filling (default: None).
        dil_its (int): Number of dilation iterations (default: 2).
    
    Returns:
        numpy.ndarray: Filled binary image.
    """
    
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
    
    """
    Perform simple threshold-based image segmentation.
    
    Args:
        im (numpy.ndarray): Input image.
        thr (float): Threshold value for segmentation.
        norm (bool): Flag to normalize the image (default: False).
        plot (bool): Flag to display a plot of the segmented image (default: False).
    
    Returns:
        numpy.ndarray: Segmented image.
    """
    
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

    """
    Make the dimensions of a 1D, 2D, or 3D array have even sizes.
    
    Args:
        mat (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Array with even dimensions.
    """
    
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

    """
    Pad two volumes with zeros to match their sizes.
    
    Args:
        vol1 (numpy.ndarray): First volume.
        vol2 (numpy.ndarray): Second volume.
        dtype (str): Data type for the padded volumes.
    
    Returns:
        numpy.ndarray: Padded vol1.
        numpy.ndarray: Padded vol2.
    """
    
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

    """
    Pad a volume with zeros to match a specific size.
    
    Args:
        vol (numpy.ndarray): Input volume.
        dims (tuple): Target dimensions for the padded volume.
        dtype (str): Data type for the padded volume.
    
    Returns:
        numpy.ndarray: Padded volume.
    """
    
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

    """
    Pad two images with zeros to match their sizes.
    
    Args:
        im1 (numpy.ndarray): First input image.
        im2 (numpy.ndarray): Second input image.
        dtype (str): Data type for the padded images.
    
    Returns:
        tuple: Padded images (im1, im2).
    """
    
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
    
    """
    Replace NaN values in an array with a specified number.
    
    Args:
        array (numpy.ndarray): Input array.
        val (float): Value to replace NaN with. Default is 0.
    
    Returns:
        numpy.ndarray: Array with NaN values replaced by the specified number.
    """
    
    return(np.where(np.isnan(array), val, array))


def number_to_nan(array, val=0):
    
    """
    Replace a specified number in an array with NaN values.
    
    Args:
        array (numpy.ndarray): Input array.
        val (float): Value to replace with NaN. Default is 0.
    
    Returns:
        numpy.ndarray: Array with the specified number replaced by NaN values.
    """
    
    array[array==val] = np.nan
    return(array)
    


def find_first_neighbors_2D(mat, r, c):
    
    """
    Find the first neighbor elements that are non-zero in a binary 2D matrix.
    
    Args:
        mat (numpy.ndarray): Binary 2D matrix.
        r (int): Row index of the element.
        c (int): Column index of the element.
    
    Returns:
        list: List of coordinates (row, column) of the first neighbor elements that are non-zero.
    """
    
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
    
def rebinmat(array, factor = 2, axis=0):
    
    """
    Rebin a 2D/3D array along an axis

    Args:
        array: numpy.ndarray.
        factor (int): Rebinning factor (default: 2).
        axis (int): Axis for rebinning (default: 0).

    Returns:
        numpy.ndarray: A rebinned array along an axis
    """

    dims = array.shape
    
    if len(dims) == 3:

        if axis == 0:    
    
            rebinned = np.zeros((int(array.shape[0]/factor), array.shape[1], array.shape[2]), dtype='float32')
            kk = 0
            for ii in tqdm(range(0, array.shape[0], factor)):
                rebinned[kk,:,:] = np.mean(array[ii:ii+factor, :,:], axis = 0)
                kk = kk + 1
        
        elif axis == 1:    
            
            rebinned = np.zeros((array.shape[0], int(array.shape[1]/2), array.shape[2]), dtype='float32')
            kk = 0
            for ii in tqdm(range(0, array.shape[1], factor)):
                rebinned[:,kk,:] = np.mean(array[:,ii:ii+factor,:], axis = 1)
                kk = kk + 1

        elif axis == 2:    
            
            rebinned = np.zeros((array.shape[0], array.shape[1], int(array.shape[2]/2)), dtype='float32')
            kk = 0
            for ii in tqdm(range(0, array.shape[2], factor)):
                rebinned[:,:,kk] = np.mean(array[:,:,ii:ii+factor], axis = 1)
                kk = kk + 1
    
    if len(dims) == 2:

        if axis == 0:    
    
            rebinned = np.zeros((int(array.shape[0]/factor), array.shape[1]), dtype='float32')
            kk = 0
            for ii in tqdm(range(0, array.shape[0], factor)):
                rebinned[kk,:] = np.mean(array[ii:ii+factor,:], axis = 0)
                kk = kk + 1
                
        elif axis == 1:    
            
            rebinned = np.zeros((array.shape[0], int(array.shape[1]/factor)), dtype='float32')
            kk = 0
            for ii in tqdm(range(0, array.shape[1], factor)):
                rebinned[:,kk] = np.mean(array[:,ii:ii+factor], axis = 1)
                kk = kk + 1

    return(rebinned)



def rebin1d(x, y, x_new):
    
    """
    Rebin 1D data from original x-axis `x` to a new x-axis `x_new` using linear interpolation
    with weighted averaging.

    This function redistributes the values in `y` (defined on axis `x`) onto a new set of 
    points `x_new` by performing distance-weighted averaging to the nearest two neighbors in `x_new`.

    For points in `x` that lie outside the bounds of `x_new`, the value of `y` is assigned 
    directly to the nearest boundary bin.

    Parameters
    ----------
    x : array-like of shape (N,)
        Original x-axis coordinates.
    y : array-like of shape (N,)
        Data values corresponding to `x`.
    x_new : array-like of shape (M,)
        New x-axis grid to which `y` values will be rebinned.

    Returns
    -------
    rebin_data : ndarray of shape (M,)
        Rebinned data values corresponding to `x_new`.

    Notes
    -----
    - The method conserves total weight by proportionally distributing each `y[i]` value
      between the two closest points in `x_new`.
    - This approach is useful for spectral regridding or when combining datasets sampled
      on different 1D grids.
    """
        
    # Arrays to store the sum of the weighted y values and the sum of the weights
    sum_weighted_y = np.zeros(len(x_new))
    sum_weights = np.zeros(len(x_new))
    
    min_x_new = np.min(x_new)
    max_x_new = np.max(x_new)
    
    # Loop over each observation point
    for i in range(len(x)):
        
        arr = np.abs(x_new - x[i])
        # Get the indices that would sort the array
        sorted_indices = np.argsort(arr)
        # Get the indices of the two smallest values
        smallest_indices = sorted_indices[:2]
            
        # print(arr)
        # print(sorted_indices)
        # print(smallest_indices)
        
        # Calculate the weights based on the relative distance to the two neighboring bins
    
        if x[i] <= min_x_new:
            
            # Add the weighted y value to the sum of weighted y values in the two bins
            sum_weighted_y[smallest_indices[0]] += y[i]
            # Add the weights to the sum of weights in the two bins
            sum_weights[smallest_indices[0]] += 1
    
        elif x[i] >= max_x_new:
            
            # Add the weighted y value to the sum of weighted y values in the two bins
            sum_weighted_y[smallest_indices[0]] += y[i]
            # Add the weights to the sum of weights in the two bins
            sum_weights[smallest_indices[0]] += 1
    
        else:
            
            if smallest_indices[1]>smallest_indices[0]:
                
                w0 = np.abs((x_new[smallest_indices[0]] - x[i]) / (x_new[smallest_indices[1]] - x_new[smallest_indices[0]]))
                w1 = 1 - w0            
                # Add the weighted y value to the sum of weighted y values in the two bins
                sum_weighted_y[smallest_indices[0]] += y[i] * w0
                sum_weighted_y[smallest_indices[1]] += y[i] * w1
                
                # Add the weights to the sum of weights in the two bins
                sum_weights[smallest_indices[0]] += w0
                sum_weights[smallest_indices[1]] += w1        
    
            elif smallest_indices[1]<smallest_indices[0]:
    
                w0 = np.abs((x_new[smallest_indices[0]] - x[i]) / (x_new[smallest_indices[1]] - x_new[smallest_indices[0]]))
                w1 = 1 - w0            
                # Add the weighted y value to the sum of weighted y values in the two bins
                sum_weighted_y[smallest_indices[0]] += y[i] * w0
                sum_weighted_y[smallest_indices[1]] += y[i] * w1
                
                # Add the weights to the sum of weights in the two bins
                sum_weights[smallest_indices[0]] += w0
                sum_weights[smallest_indices[1]] += w1    
            

    rebin_data = np.array(sum_weighted_y / sum_weights, dtype = 'float32')

    return(rebin_data)











