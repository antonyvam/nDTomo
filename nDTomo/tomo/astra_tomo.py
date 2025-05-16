# -*- coding: utf-8 -*-
"""
Tomography utility functions using the ASTRA toolbox

This module provides a collection of functions for generating sinograms, 
reconstructing 2D and 3D tomographic datasets, forward projections, and 
computing system matrices using the ASTRA toolbox. It supports both 
parallel-beam and cone-beam geometries, GPU acceleration, and multiple 
reconstruction algorithms including FBP, SIRT, and FDK.

Author: Antony Vamvakeros
"""

#%%

import os
import types
import scipy
import time
from numpy import deg2rad, arange, linspace, pi, zeros, mean, where, floor, log, inf, exp, vstack
from numpy.random import rand
from tqdm import tqdm

try:
    import astra
except ImportError:
    if os.environ.get("READTHEDOCS") == "True":
        # Mock astra module for ReadTheDocs
        astra = types.SimpleNamespace()

        def dummy(*args, **kwargs): return None
        astra.create_vol_geom = dummy
        astra.create_proj_geom = dummy
        astra.create_projector = dummy
        astra.projector = types.SimpleNamespace(matrix=dummy, delete=dummy)
        astra.data2d = types.SimpleNamespace(create=dummy, get=dummy, delete=dummy)
        astra.data3d = types.SimpleNamespace(create=dummy, get=dummy, delete=dummy)
        astra.algorithm = types.SimpleNamespace(create=dummy, run=dummy, delete=dummy)
        astra.astra_dict = dummy
        astra.astra = types.SimpleNamespace(set_gpu_index=dummy, delete=dummy)
        astra.geom_postalignment = dummy
        astra.creators = types.SimpleNamespace(create_vol_geom=dummy)
        astra.create_sino = lambda x, y: (None, x)
        astra.create_sino3d_gpu = lambda x, y, z: (None, x)
    else:
        raise
    
def astra_Amatrix(ntr, ang):

    """
    Generate a sparse system matrix A for 2D parallel-beam geometry using ASTRA.

    Parameters
    ----------
    ntr : int
        Number of translation steps (image size).
    ang : array_like
        Array of projection angles in radians.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse system matrix (size: [n_projections * detector_pixels, ntr^2]).
    """
    vol_geom = astra.create_vol_geom(ntr, ntr) 
    proj_geom = astra.create_proj_geom('parallel', 1.0, ntr, ang) 
    proj_id = astra.create_projector('line', proj_geom, vol_geom) 
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id)
    A = scipy.sparse.csr_matrix.astype(A, dtype = 'float32')

    astra.projector.delete(proj_id)
    astra.data2d.delete(matrix_id)

    return(A)

def astra_rec_single(sino, theta=None, scanrange = '180', method='FBP_CUDA', filt='Ram-Lak', nits = None, timing=False):
    
    """
    Reconstruct a single 2D slice from a sinogram using ASTRA Toolbox.

    Parameters
    ----------
    sino : ndarray
        2D sinogram (shape: [translation_steps, projections]).
    theta : ndarray, optional
        Projection angles in radians. If None, generated from scanrange.
    scanrange : str, optional
        Angle range ('180' or '360') to auto-generate theta if not provided.
    method : str, optional
        ASTRA reconstruction method, e.g., 'FBP_CUDA', 'SIRT', 'CGLS'.
    filt : str, optional
        Filter type for FBP-based methods (e.g., 'Ram-Lak', 'Hann').
    nits : int, optional
        Number of iterations (for iterative methods).
    timing : bool, optional
        Print reconstruction time if True.

    Returns
    -------
    rec : ndarray
        Reconstructed 2D image.
    """
    
    npr = sino.shape[1] # Number of projections
    
    if theta is None:
        if scanrange == '180':
            theta = deg2rad(arange(0, 180, 180/npr))
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr))
            
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(sino.shape[0], sino.shape[0])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
    proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*sino.shape[0]), theta)
    # Create a sinogram using the GPU.
    proj_id = astra.create_projector('strip',proj_geom,vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sino.transpose())
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    if method == 'FBP' or method == 'FBP_CUDA':
        cfg['option'] = { 'FilterType': filt }
    else:
        if method == 'SART' or method == 'SIRT' or method == 'SART_CUDA' or method == 'SIRT_CUDA' or method == 'ART':
            cfg['option']={}
            cfg['option']['MinConstraint'] = 0
        if nits is None:
            nits = 10 
    
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    if timing:
        start=time.time()

    if method == 'FBP' or method == 'FBP_CUDA':
        rec = astra.algorithm.run(alg_id)
    else:
        rec = astra.algorithm.run(alg_id, nits)
    
    # Get the result
    
    rec = astra.data2d.get(rec_id)
    if timing:
        print((time.time()-start))
        
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    
    return(rec)


def astra_create_sino(im, npr = None, scanrange = '180', theta=None, proj_id=None):
    
    """
    Generate a 2D sinogram from an image using ASTRA Toolbox.

    Parameters
    ----------
    im : ndarray
        2D input image.
    npr : int, optional
        Number of projection angles. Defaults to image width.
    scanrange : str, optional
        Range of scan angles ('180' or '360') if theta is not provided.
    theta : ndarray, optional
        Projection angles in radians. Overrides scanrange if provided.
    proj_id : int, optional
        Pre-computed projector ID. If None, will be created.

    Returns
    -------
    sinogram : ndarray
        2D sinogram with shape (n_angles, n_pixels).
    """

    if npr is None:
        npr = im.shape[0]

    if theta is None:
        if scanrange == '180':
            theta = deg2rad(arange(0, 180, 180/npr))
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr))

    if proj_id is None:
        proj_id = astra_create_sino_geo(im, theta)

    sinogram_id, sinogram = astra.create_sino(im, proj_id)

    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return(sinogram)

def astra_create_sino_geo(im, theta=None):
    
    """
    Create an ASTRA projector ID for 2D parallel beam geometry.

    Parameters
    ----------
    im : ndarray
        2D image array.
    theta : ndarray, optional
        Array of projection angles in radians. If None, defaults to linspace(0, pi, im.shape[0]).

    Returns
    -------
    proj_id : int
        ASTRA projector ID for the given geometry.
    """    
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(im.shape[0], im.shape[0])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and
    # im.shape[0] detector pixels of width 1.
    # For more details on available geometries, see the online help of the
    # function astra_create_proj_geom .
    if theta is None:
        theta = linspace(0,pi,im.shape[0])

    proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[0], theta, False)
    # Create a sinogram using the GPU.
    # Note that the first time the GPU is accessed, there may be a delay
    # of up to 10 seconds for initialization.
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)        
    
    return(proj_id)

            
def astra_create_sinostack(vol, npr = None, scanrange = '180', theta=None, proj_id=None, dtype='float32'):

    """
    Create a sinogram stack from a 3D volume using ASTRA.

    Parameters
    ----------
    vol : ndarray
        3D volume (shape: [N, N, num_slices]).
    npr : int, optional
        Number of projection angles. Defaults to volume width.
    scanrange : str, optional
        Scan angle range ('180' or '360').
    theta : ndarray, optional
        Projection angles in radians.
    proj_id : int, optional
        Reuse existing ASTRA projector ID.
    dtype : str, optional
        Data type for output (default: 'float32').

    Returns
    -------
    sinograms : ndarray
        Sinogram stack with shape (n_angles, N, num_slices).
    """
    
    if npr is None:
        npr = vol.shape[0]

    if theta is None:
        if scanrange == '180':
            theta = deg2rad(arange(0, 180, 180/npr))
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr))

    if proj_id is None:
        proj_id = astra_create_sino_geo(vol[:,:,0], theta)

    sz = vol.shape[0]
    nim = vol.shape[2]

    sinograms = zeros((len(theta), sz, nim), dtype=dtype)
    
    for ii in tqdm(range(nim)):
        
         sinogram_id, sinograms[:,:,ii] = astra.create_sino(vol[:,:,ii], proj_id)
                      
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return(sinograms)

def astra_create_geo(sino, theta=None):
        
    """
    Set up volume and projection geometry and create a reconstruction volume ID.

    Parameters
    ----------
    sino : ndarray
        2D sinogram array (shape: [translations, angles]).
    theta : ndarray, optional
        Array of angles in radians.

    Returns
    -------
    proj_geom : dict
        ASTRA projection geometry.
    rec_id : int
        ASTRA volume ID.
    proj_id : int
        ASTRA projector ID.
    """
    
    npr = sino.shape[1] # Number of projections
    
    if theta is None:
        theta = deg2rad(arange(0, 180, 180/npr))
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(sino.shape[0], sino.shape[0])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
    proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*sino.shape[0]), theta)
    # Create a sinogram using the GPU.
    proj_id = astra.create_projector('strip',proj_geom,vol_geom)
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
        
    return(proj_geom, rec_id, proj_id)
             
             
def astra_rec_alg(sino, proj_geom, rec_id, proj_id, method='FBP', filt='Ram-Lak'):

    """
    Reconstruct a 2D image from a sinogram using the specified ASTRA algorithm.

    Parameters
    ----------
    sino : ndarray
        2D sinogram.
    proj_geom : dict
        ASTRA projection geometry.
    rec_id : int
        ASTRA volume ID for output.
    proj_id : int
        ASTRA projector ID.
    method : str, optional
        Reconstruction algorithm ('FBP', 'SIRT', etc.).
    filt : str, optional
        Filter type (only for FBP).

    Returns
    -------
    rec : ndarray
        2D reconstructed image.
    """

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino.transpose())
        
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    
    if method == 'FBP':
        cfg['option'] = { 'FilterType': filt }
    
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    # Get the result
    start=time.time()
    rec = astra.data2d.get(rec_id)
    print((time.time()-start))             
    
    astra.data2d.delete(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return(rec)
                   
             
def astra_rec_vol(sinos, scanrange = '180', theta=None,  proj_geom=None, proj_id=None, rec_id=None, method='FBP_CUDA', filt='Ram-Lak', pbar = True):

    """
    Reconstruct a 3D volume from a stack of sinograms using ASTRA.

    Parameters
    ----------
    sinos : ndarray
        3D stack of sinograms (shape: [N, angles, slices]).
    scanrange : str, optional
        Scan angle range ('180' or '360').
    theta : ndarray, optional
        Array of angles in radians.
    proj_geom : dict, optional
        Predefined projection geometry.
    proj_id : int, optional
        Predefined projector ID.
    rec_id : int, optional
        Predefined reconstruction volume ID.
    method : str, optional
        Reconstruction algorithm (default: 'FBP_CUDA').
    filt : str, optional
        Filter type.
    pbar : bool, optional
        Show progress bar.

    Returns
    -------
    rec : ndarray
        3D reconstructed volume.
    """
        
    npr = sinos.shape[1] # Number of projections
    
    if theta is None:
        if scanrange == '180':
            theta = deg2rad(arange(0, 180, 180/npr))
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr))
    
    if proj_geom is None:
        # Create a basic square volume geometry
        vol_geom = astra.create_vol_geom(sinos.shape[0], sinos.shape[0])
        # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
        proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*sinos.shape[0]), theta)
    if proj_geom is None:
        if method == 'FBP_CUDA':
            # Create a sinogram using the GPU. 
            proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
        elif method == 'FBP':
            # Create a sinogram using the GPU. 
            proj_id = astra.create_projector('strip',proj_geom,vol_geom)
    if rec_id is None:
        # Create a data object for the reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom)    
    
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = proj_id
    if method == 'FBP' or method == 'FBP_CUDA':
        cfg['option'] = { 'FilterType': filt }    
    
    loop_range = tqdm(range(sinos.shape[2])) if pbar else range(sinos.shape[2])

    rec = zeros((sinos.shape[0], sinos.shape[0], sinos.shape[2]), dtype='float32')
    for ii in loop_range:

        sinogram_id = astra.data2d.create('-sino', proj_geom, sinos[:,:,ii].transpose())
        
        cfg['ProjectionDataId'] = sinogram_id
        
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get the result
        rec[:,:,ii] = astra.data2d.get(rec_id)
             
        astra.algorithm.delete(alg_id)
        
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return(rec)

def astra_rec_2vols(sinos, method='FBP_CUDA', filt='Ram-Lak'):
     
    """
    Reconstruct two interleaved volumes from alternating angle subsets.

    Parameters
    ----------
    sinos : ndarray
        3D array with shape (z, projections, x).
    method : str, optional
        Reconstruction method (default: 'FBP_CUDA').
    filt : str, optional
        Filter type (default: 'Ram-Lak').

    Returns
    -------
    vol1 : ndarray
        Volume reconstructed from even-indexed angles.
    vol2 : ndarray
        Volume reconstructed from odd-indexed angles.
    """
    
    npr = sinos.shape[1] # Number of projections
    
    theta = deg2rad(arange(0, 180, 180/npr))
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(sinos.shape[2], sinos.shape[2])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
    proj_geom1 = astra.create_proj_geom('parallel', 1.0, int(1.0*sinos.shape[2]), theta[0::2])
    proj_geom2 = astra.create_proj_geom('parallel', 1.0, int(1.0*sinos.shape[2]), theta[1::2])
    # Create a sinogram using the GPU. 
    proj_id1 = astra.create_projector('strip',proj_geom1,vol_geom)
    proj_id2 = astra.create_projector('strip',proj_geom2,vol_geom)
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    
    cfg['option'] = { 'FilterType': 'Ram-Lak' }  
    
    vol1 = zeros((sinos.shape[2], sinos.shape[2], sinos.shape[0]))
    vol2 = zeros((sinos.shape[2], sinos.shape[2], sinos.shape[0]))
    
    for ii in tqdm(range(sinos.shape[0])):
    
        s = sinos[ii,:,:].transpose()
    
        sinogram_id = astra.data2d.create('-sino', proj_geom1, s[:,0::2].transpose())
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id1
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        vol1[:,:,ii] = astra.data2d.get(rec_id)
        astra.algorithm.delete(alg_id)
        
        sinogram_id = astra.data2d.create('-sino', proj_geom2, s[:,1::2].transpose())
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id2
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        vol2[:,:,ii] = astra.data2d.get(rec_id)
        astra.algorithm.delete(alg_id)
    
    astra.projector.delete(proj_id1)
    astra.projector.delete(proj_id2)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return(vol1, vol2)

def astra_rec_vol_singlesino(sino, ims = 100, ofs=None, scanrange = '180', proj_geom=None, proj_id=None, rec_id=None, method='FBP_CUDA', filt='Ram-Lak'):

    """
    Reconstruct multiple volumes from a single sinogram using random angular offsets and interleaved angle subsets.

    This function generates two 3D volumes by randomly rotating the full-angle set (via `ofs`)
    and reconstructing from interleaved projections (even vs odd indices) at each iteration.

    Parameters
    ----------
    sino : ndarray
        2D sinogram (shape: [translation_steps, projections]).
    ims : int, optional
        Number of volumes to reconstruct (default: 100).
    ofs : ndarray, optional
        Random angular offsets (length: ims). If None, generated uniformly in [0,1).
    scanrange : str, optional
        '180' or '360' scan in degrees (default: '180').
    proj_geom : dict, optional
        ASTRA projection geometry, will be overwritten inside loop.
    proj_id : int, optional
        ASTRA projector ID (ignored, recomputed each iteration).
    rec_id : int, optional
        ASTRA reconstruction volume ID.
    method : str, optional
        ASTRA reconstruction algorithm (default: 'FBP_CUDA').
    filt : str, optional
        Filter type for FBP methods (default: 'Ram-Lak').

    Returns
    -------
    rec : ndarray
        Reconstructed volumes from even-indexed angles (shape: [H, W, ims]).
    rec2 : ndarray
        Reconstructed volumes from odd-indexed angles (shape: [H, W, ims]).
    """
        
    sino1 = sino[:,0::2]
    sino2 = sino[:,1::2]
    
    if ofs is None:
        ofs = rand(ims)
    
    npr = sino.shape[1] # Number of projections
    if proj_geom is None:
        # Create a basic square volume geometry
        vol_geom = astra.create_vol_geom(sino.shape[0], sino.shape[0])
        # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
        
    if rec_id is None:
        # Create a data object for the reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom)   
        
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    if method == 'FBP' or method == 'FBP_CUDA':
        cfg['option'] = { 'FilterType': filt }           
        
    rec = zeros((sino.shape[0], sino.shape[0], ims))
    rec2 = zeros((sino.shape[0], sino.shape[0], ims))
    
    for ii in tqdm(range(ims)):

        if scanrange == '180':
            theta = deg2rad(arange(0, 180, 180/npr)) + ofs[ii]*360
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr)) + ofs[ii]*360
            
        proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*sino.shape[0]), theta[0::2])
        proj_geom2 = astra.create_proj_geom('parallel', 1.0, int(1.0*sino.shape[0]), theta[1::2])
        

        if method == 'FBP_CUDA':
            # Create a sinogram using the GPU. 
            proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
            proj_id2 = astra.create_projector('cuda',proj_geom2,vol_geom)
        elif method == 'FBP':
            # Create a sinogram using the GPU. 
            proj_id = astra.create_projector('strip',proj_geom,vol_geom)
            proj_id2 = astra.create_projector('strip',proj_geom2,vol_geom)
             
        sinogram_id = astra.data2d.create('-sino', proj_geom, sino1.transpose())
        sinogram_id2 = astra.data2d.create('-sino', proj_geom2, sino2.transpose())

        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sinogram_id
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        rec[:,:,ii] = astra.data2d.get(rec_id)

        cfg['ProjectorId'] = proj_id2
        cfg['ProjectionDataId'] = sinogram_id2
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        rec2[:,:,ii] = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return(rec, rec2)

def ConeBeamCTGeometry(downSizeFactor=4, distance_source_detector=926.79, distance_source_origin=349.565,
                  detector_pixel_size = 0.1, mag_factor = 2.65, horizontalOffset=0, verticalOffset=0):

    """
    Define cone-beam CT geometry.

    Parameters
    ----------
    downSizeFactor : int, optional
        Downsampling factor for resolution.
    distance_source_detector : float
        Distance from source to detector [mm].
    distance_source_origin : float
        Distance from source to rotation center [mm].
    detector_pixel_size : float
        Pixel size of detector [mm].
    mag_factor : float
        Magnification factor.
    horizontalOffset : float
        Horizontal offset in pixels.
    verticalOffset : float
        Vertical offset in pixels.

    Returns
    -------
    geo : dict
        Geometry dictionary.
    """
   
    geo = {}
    geo['distance_source_detector'] =  distance_source_detector
    geo['distance_source_origin'] =  distance_source_origin
    geo['downSizeFactor'] = downSizeFactor
    geo['detector_pixel_size'] = detector_pixel_size*downSizeFactor
    geo['rec_pixelSize'] = detector_pixel_size/mag_factor/10*downSizeFactor # in cm
    geo['horizontalOffset'] = 0 # in pixels
    geo['verticalOffset'] = 0 # in pixels
    return(geo)

def ConeBeamCTbeam(sf = 15, whiteCounts = 4500, countsToCut = 1):
    
    """
    Define beam parameters for cone-beam CT simulation.

    Parameters
    ----------
    sf : float
        Scaling factor for attenuation.
    whiteCounts : float
        Maximum detector counts.
    countsToCut : float
        Threshold for zeroing low-intensity counts.

    Returns
    -------
    beam : dict
        Beam parameter dictionary.
    """

    beam = {}
    beam['sf'] = sf
    beam['whiteCounts'] = whiteCounts
    beam['countsToCut'] = countsToCut
    return(beam)

def coneBeamFP(vol, nproj, geo, v_cut = 0, detector_cols=None, detector_rows=None):    
        
    """
    Perform forward projection of a 3D volume using cone-beam geometry (ASTRA GPU-based).

    Projects the input volume to generate synthetic cone-beam radiographs over a full 360° rotation.

    Parameters
    ----------
    vol : ndarray
        3D volume to project (shape: [detector_rows, _, detector_cols]).
    nproj : int
        Number of projection angles (evenly spaced over 0–2π).
    geo : dict
        Geometry dictionary (as returned by ConeBeamCTGeometry()).
    v_cut : float, optional
        Fraction of the volume to crop from the top (default: 0).
    detector_cols : int, optional
        Number of horizontal detector pixels. If None, inferred from vol.
    detector_rows : int, optional
        Number of vertical detector pixels. If None, inferred from vol.

    Returns
    -------
    proj_data : ndarray
        3D array of cone-beam projections (shape: [detector_rows, nproj, detector_cols]).
    """
        
    v_cut_local = int(v_cut * 8 / geo['downSizeFactor'])
    v_window = range(v_cut_local,  vol.shape[0])    
    
    v_corr = v_window[0] / 2

    # Configuration. - need to read this from a configuration file
    distance_origin_detector =geo['distance_source_detector'] - geo['distance_source_origin']  # [mm]
    if detector_cols is None:
        detector_cols = vol.shape[2]  # Horizontal size of detector [pixels].
    if detector_rows is None:
        detector_rows = vol.shape[0]  # Vertical size of detector [pixels].
    angles = linspace(0, 2 * pi, num=nproj, endpoint=False)
 
    # Set up multi-GPU usage.
    # This only works for 3D GPU forward projection and back projection.
    astra.astra.set_gpu_index([0, 1])
    
    # Create geometry
    proj_geom = \
        astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                               (geo['distance_source_origin'] + distance_origin_detector) /
                               geo['detector_pixel_size'], 0)

    # Center-of-rotation correction (by 0 pixels horizontally)
    proj_geom_cor = astra.geom_postalignment(proj_geom, \
        [geo['horizontalOffset'], v_corr+geo['verticalOffset']])


    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)

    proj_id, proj_data =  astra.create_sino3d_gpu(vol, proj_geom_cor, vol_geom)
        
    astra.astra.delete(proj_id)
    
    return(proj_data)
    
    
def radiographs_mu(projections, geo, beam):

    """
    Convert line integrals to intensity using beam attenuation, then re-log.

    Parameters
    ----------
    projections : ndarray
        Projection data (line integrals).
    geo : dict
        CT geometry.
    beam : dict
        Beam parameters.

    Returns
    -------
    P : ndarray
        Log-normalized projection data.
    """
    
    I = beam['whiteCounts'] * exp(-projections*beam['sf']*geo['rec_pixelSize'])
    I =  where(I > beam['countsToCut'], I, 0)
    I = floor(I)
    P = -log(I*1/beam['whiteCounts'])*1/geo['rec_pixelSize']
    P =  where(P == inf, 0, P)
    return(P)

def coneBeamFDK(projections, geo, v_cut = 0):    
   
    """
    Reconstruct a 3D volume from cone-beam projections using FDK algorithm.

    Parameters
    ----------
    projections : ndarray
        3D cone-beam sinograms (shape: [detector_rows, angles, detector_cols]).
    geo : dict
        Geometry dictionary returned by ConeBeamCTGeometry().
    v_cut : float, optional
        Vertical cropping factor (fraction from top, default: 0).

    Returns
    -------
    reconstruction : ndarray
        3D reconstructed volume (ASTRA volume geometry).
    """

    v_cut_local = int(v_cut * 8 / geo['downSizeFactor'])
    v_window = range(v_cut_local,  projections.shape[0])    
    v_corr = v_window[0] / 2

    # Configuration. - need to read this from a configuration file
    distance_origin_detector =geo['distance_source_detector'] - geo['distance_source_origin']  # [mm]
    detector_cols = projections.shape[2]  # Horizontal size of detector [pixels].
    detector_rows = projections.shape[0]  # Vertical size of detector [pixels].
    num_of_projections= projections.shape[1]
    angles = linspace(0, 2 * pi, num=num_of_projections, endpoint=False)
 
    # Set up multi-GPU usage.
    # This only works for 3D GPU forward projection and back projection.
    astra.astra.set_gpu_index([0, 1])
    
    # Create geometry
    proj_geom = \
        astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                               ( geo['distance_source_origin'] + distance_origin_detector) /
                               geo['detector_pixel_size'], 0)

    # Center-of-rotation correction (by 0 pixels horizontally)
    proj_geom_cor = astra.geom_postalignment(proj_geom, \
        [geo['horizontalOffset'], v_corr+geo['verticalOffset']])

    # Create astra projection set
    projections_id = astra.data3d.create('-sino', proj_geom_cor, projections)

    # Create reconstruction.
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

    recon_method = 'FDK_CUDA'
    alg_cfg = astra.astra_dict(recon_method)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)

    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)

    astra.astra.delete(projections_id)
    astra.astra.delete(reconstruction_id)

    return(reconstruction)

def coneBeamFP_FDK(vol, nproj, geo, beam=None, v_cut = 0):    
    
    """
    Perform forward projection of a 3D volume and reconstruct it using FDK.

    Optionally applies exponential attenuation and Poisson noise before reconstruction.

    Parameters
    ----------
    vol : ndarray
        3D volume to be projected (shape: [H, W, D]).
    nproj : int
        Number of projection angles.
    geo : dict
        Cone-beam CT geometry.
    beam : dict, optional
        Beam model for forward projection (see ConeBeamCTbeam()).
    v_cut : float, optional
        Fraction of top pixels to crop from vertical axis.

    Returns
    -------
    reconstruction : ndarray
        3D volume reconstructed from forward projections using FDK.
    """
    
    v_cut_local = int(v_cut * 8 / geo['downSizeFactor'])
    v_window = range(v_cut_local,  vol.shape[0])    
    
    v_corr = v_window[0] / 2

    # Configuration. - need to read this from a configuration file
    distance_origin_detector =geo['distance_source_detector'] - geo['distance_source_origin']  # [mm]
    detector_cols = vol.shape[2]  # Horizontal size of detector [pixels].
    detector_rows = vol.shape[0]  # Vertical size of detector [pixels].
    angles = linspace(0, 2 * pi, num=nproj, endpoint=False)
 
    # Set up multi-GPU usage.
    # This only works for 3D GPU forward projection and back projection.
    astra.astra.set_gpu_index([0, 1])
    
    # Create geometry
    proj_geom = \
        astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                               (geo['distance_source_origin'] + distance_origin_detector) /
                               geo['detector_pixel_size'], 0)

    # Center-of-rotation correction (by 0 pixels horizontally)
    proj_geom_cor = astra.geom_postalignment(proj_geom, \
        [geo['horizontalOffset'], v_corr+geo['verticalOffset']])


    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)

    proj_id, proj_data =  astra.create_sino3d_gpu(vol, proj_geom_cor, vol_geom)
        
    if beam is not None:
        proj_data = radiographs_mu(proj_data, geo, beam)
     
    # Set up multi-GPU usage.
    # This only works for 3D GPU forward projection and back projection.
    astra.astra.set_gpu_index([0, 1])
    
    # Create astra projection set
    projections_id = astra.data3d.create('-sino', proj_geom_cor, proj_data)

    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

    recon_method = 'FDK_CUDA'
    alg_cfg = astra.astra_dict(recon_method)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)

    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)

    astra.astra.delete(algorithm_id)
    astra.astra.delete(proj_id)
    astra.astra.delete(projections_id)
    astra.astra.delete(reconstruction_id)
    
    return(reconstruction)


def create_Amat(npix, ang):
    
    """
    Construct a system matrix A using ASTRA's sparse matrix projector.

    Parameters
    ----------
    npix : int
        Number of pixels in one dimension (volume assumed square).
    ang : ndarray
        Projection angles in radians.

    Returns
    -------
    Acoo : scipy.sparse.coo_matrix
        Sparse matrix in coordinate format.
    values : ndarray
        Non-zero values of the sparse matrix.
    indices : ndarray
        Row and column indices of non-zero elements (shape: [2, nnz]).
    shape : tuple
        Shape of the full matrix.
    """
    
    vol_geom = astra.create_vol_geom(npix, npix) 
    proj_geom = astra.create_proj_geom('parallel', 1.0, int(npix), ang) 
    proj_id = astra.create_projector('strip', proj_geom, vol_geom) 
    
    # matrix id
    matrix_id = astra.projector.matrix(proj_id)
    
    # Get the projection matrix as a Scipy sparse matrix.
    A = astra.matrix.get(matrix_id)
    
    # Convert it to float32
    A = A.astype(dtype='float32')
    
    print(A.shape)
    
    Acoo = A.tocoo()
    values = Acoo.data
    indices = vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    
    return(Acoo, values, indices, shape)
    

