# -*- coding: utf-8 -*-
"""
Functions for tomography using astra-toolbox library

@author: Antony Vamvakeros
"""


import scipy, astra, time
from numpy import deg2rad, arange, linspace, pi, zeros, mod, mean, where, floor, log, inf, exp
from numpy.random import rand
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

def astra_rec_single(sino, theta=None, scanrange = '180', method='FBP_CUDA', filt='Ram-Lak', nits = None):
    
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

    start=time.time()

    if method == 'FBP' or method == 'FBP_CUDA':
        rec = astra.algorithm.run(alg_id)
    else:
        rec = astra.algorithm.run(alg_id, nits)
    
    # Get the result
    
    rec = astra.data2d.get(rec_id)
    
    print((time.time()-start))
        
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    
    return(rec)


def astra_create_sino(im, npr = None, scanrange = '180', theta=None, proj_id=None):
    
    '''
    Create a sinogram using the astra toolbox for parallel beam geometry
    '''

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

            
def astra_create_sinostack(vol, npr = None, scanrange = '180', theta=None, proj_id=None):

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

    sinograms = zeros((len(theta), sz, nim))
    
    for ii in tqdm(range(nim)):
        
         sinogram_id, sinograms[:,:,ii] = astra.create_sino(vol[:,:,ii], proj_id)
                      
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return(sinograms)

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
             
             
def astra_rec_alg(sino, proj_geom, rec_id, proj_id, method='FBP', filt='Ram-Lak'):

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
    
    astra.data2d.delete(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return(rec)
                   
             
def astra_rec_vol(sinos, scanrange = '180', theta=None,  proj_geom=None, proj_id=None, rec_id=None, method='FBP_CUDA', filt='Ram-Lak'):

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
    
    rec = zeros((sinos.shape[0], sinos.shape[0], sinos.shape[2]))
    for ii in tqdm(range(sinos.shape[2])):

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
     
    '''
    Inputs:
        s: (z, proj, x) 
    '''
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
        s = s - mean(s[0,:])
        s = where(s<0, 0, s)
        s = s[:-13,:]
        # s = s.astype('float32')
    
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

def astra_rec_vol_singlesino(sino, ims = 100, scanrange = '180', proj_geom=None, proj_id=None, rec_id=None, method='FBP_CUDA', filt='Ram-Lak'):

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
        
    sino1 = sino[:,0::2]
    sino2 = sino[:,1::2]
    
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
            theta = deg2rad(arange(0, 180, 180/npr)) + rand(1)*360
        elif scanrange == '360':
            theta = deg2rad(arange(0, 360, 360/npr)) + rand(1)*360
            
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
    
    beam = {}
    beam['sf'] = 15
    beam['whiteCounts'] = 4500
    beam['countsToCut'] = 1
    return(beam)

def coneBeamFP(vol, nproj, geo, v_cut = 0):    
        
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
        
    astra.astra.delete(proj_id)
    
    return(proj_data)
    
    
def radiographs_mu(projections, geo, beam):
    
    I = beam['whiteCounts'] * exp(-projections*beam['sf']*geo['rec_pixelSize'])
    I =  where(I > beam['countsToCut'], I, 0)
    I = floor(I)
    P = -log(I*1/beam['whiteCounts'])*1/geo['rec_pixelSize']
    P =  where(P == inf, 0, P)
    return(P)

def coneBeamFDK(projections, geo, v_cut = 0):    
   
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

def coneBeamFP_FDK(vol, nproj, geo, beam, v_cut = 0):    
        
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
