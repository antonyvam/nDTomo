# -*- coding: utf-8 -*-
"""
Tensorflow functions for tomography

@author: Antony Vamvakeros
"""


import scipy, astra, time
from numpy import deg2rad, arange, linspace, pi, zeros, mod
from tqdm import tqdm

def astra_Amatrix(ntr, ang):

    '''
    Create A matrix using the astra toolbox
    Might need to delete extra stuff

    Be careful how you define the projection angles
    Example:
    npr = 180
    theta = np.arange(0, 180, 180/npr)
    ang = np.radians(theta)
    '''
    vol_geom = astra.create_vol_geom(ntr, ntr) 
    proj_geom = astra.create_proj_geom('parallel', 1.0, ntr, ang) 
    proj_id = astra.create_projector('line', proj_geom, vol_geom) 
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id)
    A = scipy.sparse.csr_matrix.astype(A, dtype = 'float32')

    astra.projector.delete(proj_id)
    astra.data2d.delete(matrix_id)

    return(A)

def astra_rec_single(sino, theta=None, method='FBP', filt='Ram-Lak'):
    
    '''
    2D ct reconstruction using the astra-toolbox
    1st dim in sinogram is translation steps, 2nd is projections

    Available astra-toolbox reconstruction algorithms:
    ART, SART, SIRT, CGLS, FBP
    SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA
    
    possible values for FilterType:
    none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    blackman-nuttall, flat-top, kaiser, parzen    
    '''
    
    
    npr = sino.shape[1] # Number of projections
    
    if theta is None:
        theta = deg2rad(arange(0, 180, 180/npr))
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(sino.shape[0], sino.shape[0])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and image.shape[0] detector pixels of width 1.
    proj_geom = astra.create_proj_geom('parallel', 1.0, int(1.0*sino.shape[0]), theta)
    # Create a sinogram using the GPU.
    proj_id = astra.create_projector('strip',proj_geom,vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sino.transpose())
    
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    
    cfg = astra.astra_dict('FBP')
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
        
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    
    return(rec)


def astra_create_sino_geo(im, theta=None):
    
    # Create a basic square volume geometry
    vol_geom = astra.create_vol_geom(im.shape[0], im.shape[0])
    # Create a parallel beam geometry with 180 angles between 0 and pi, and
    # im.shape[0] detector pixels of width 1.
    # For more details on available geometries, see the online help of the
    # function astra_create_proj_geom .
    if theta == None:
        theta = linspace(0,pi,im.shape[0])
    proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[0], theta, False)
    # Create a sinogram using the GPU.
    # Note that the first time the GPU is accessed, there may be a delay
    # of up to 10 seconds for initialization.
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)        
    
    return(proj_id)

def astra_create_sino(im, proj_id):
    
    sinogram_id, sinogram = astra.create_sino(im, proj_id)
            
    return(sinogram)
            
def astra_create_sinostack(vol, sz, theta, nim, proj_id):

    sinograms = zeros((sz, len(theta), nim))
    
    for ii in range(nim):
        
         sinogram_id, sinograms[:,:,ii] = astra.create_sino(vol[:,:,ii], proj_id)
         
         if mod(ii, 1000) == 0:
             print('Sinogram %d out of %d' %(ii, nim))
             

def astra_create_geo(sino, theta=None):
    
    '''
    2D ct reconstruction using the astra-toolbox
    1st dim in sinogram is translation steps, 2nd is projections
    '''
    
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
             
             
def astre_rec_alg(sino, proj_geom, rec_id, proj_id, method='FBP', filt='Ram-Lak'):

    '''
    Reconstruct a single sinogram with astra toolbox
    
    Available astra-toolbox reconstruction algorithms:
    ART, SART, SIRT, CGLS, FBP
    SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA
    
    possible values for FilterType:
    none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    blackman-nuttall, flat-top, kaiser, parzen        
    '''    

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
             
    return(rec)
             
             
             
def astre_rec_vol(sinos, proj_geom, rec_id, proj_id, method='FBP', filt='Ram-Lak'):

    '''
    Reconstruct a sinogram volume with astra toolbox
    
    Available astra-toolbox reconstruction algorithms:
    ART, SART, SIRT, CGLS, FBP
    SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA
    
    possible values for FilterType:
    none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    blackman-nuttall, flat-top, kaiser, parzen    
    '''    
    
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = proj_id
    if method == 'FBP':
        cfg['option'] = { 'FilterType': filt }    
    
    rec = zeros((sinos.shape[0], sinos.shape[0], sinos.shape[2]))
    for ii in tqdm(range(sinos.shape[2])):

        sinogram_id = astra.data2d.create('-sino', proj_geom, sinos[:,:,ii].transpose())
        
        cfg['ProjectionDataId'] = sinogram_id
        
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get the result
        rec[:,:,ii] = astra.data2d.get(rec_id)
             
    return(rec)
                      
             
             
             
             
             
             