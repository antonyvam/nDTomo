# -*- coding: utf-8 -*-
"""
Create phantoms (2D-5D) for image processing, tomography, and spectral imaging experiments.

This module provides functions for generating synthetic image phantoms 
to support the development, testing, and validation of image processing, 
computer vision, and tomography algorithms. The generated datasets can 
simulate various spatial, spectral, and temporal patterns for use in 
machine learning, reconstruction, and segmentation studies.

Phantom types include:
- **2D phantoms:** Geometric shapes, random patterns, Shepp-Logan phantom, and structured test patterns.
- **3D phantoms:** Volumetric datasets with controlled spatial distributions.
- **4D phantoms:** Datasets with an additional spectral or temporal dimension.
- **5D phantoms:** Fully dynamic datasets with spatial, spectral, and temporal variations.

The module includes utilities for defining spatial distributions, 
assigning spectral signatures, and simulating motion across multiple 
dimensions. It also provides options for randomization, normalization, 
and controlled placement of features.

@author: Dr A. Vamvakeros
"""

from nDTomo.methods.misc import ndtomopath, cirmask
from xdesign import Mesh, SiemensStar, Circle, Triangle, DogaCircles, Phantom, Point, SimpleMaterial, discrete_phantom, SlantedSquares
import numpy as np
import h5py
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from skimage.draw import random_shapes
from scipy.interpolate import interp1d
from tqdm import tqdm

def SheppLogan(npix):

    """
    Generate a Shepp-Logan phantom using scikit-image.

    Parameters
    ----------
    npix : int
        Desired output image size (npix x npix).

    Returns
    -------
    np.ndarray
        2D Shepp-Logan phantom image scaled to the specified size.
    """

    im = shepp_logan_phantom()
    im = rescale(im, scale=npix/im.shape[0], mode='reflect')
    return(im)

def phantom_random_shapes(sz=368, min_shapes=3, max_shapes=10, 
                          min_size=5, max_size=50, shape=None, 
                          norm=False, allow_overlap=True):

    """
    Generate an image containing randomly placed geometric shapes using skimage.draw

    Parameters
    ----------
    sz : int, optional
        Size of the output square image (default is 368x368).
    min_shapes : int, optional
        Minimum number of shapes to generate (default is 3).
    max_shapes : int, optional
        Maximum number of shapes to generate (default is 10).
    min_size : int, optional
        Minimum size of shapes in pixels (default is 5).
    max_size : int, optional
        Maximum size of shapes in pixels (default is 50).
    shape : str or list of str, optional
        Type of shapes to include. Options: 'rectangle', 'circle', 'triangle', 'ellipse'.
        If None, all shape types are allowed (default is None).
    norm : bool, optional
        Whether to normalize the image so that values are in the range [0, 1] (default is False).
    allow_overlap : bool, optional
        Whether shapes are allowed to overlap (default is True).

    Returns
    -------
    np.ndarray
        2D image array containing the generated shapes.
    """

    im, _ = random_shapes((sz, sz), min_shapes=min_shapes, max_shapes=max_shapes, 
                          shape=shape, multichannel=False, min_size=min_size, 
                          max_size=max_size, allow_overlap=allow_overlap)

    # Convert background pixels (255) to 1, and all shape pixels to 0-1 range
    im = np.where(im == 255, 1, im) - 1

    if norm:
        im = im / np.max(im)

    return im

def phantom_comp_im(npix=512, nshapes=50, minsize=5, maxsize=50, shape=None, 
                    norm=False, allow_overlap=True, tomo=False, SL=False):
    """
    Generate a composite phantom image containing multiple randomly shaped objects.

    The generated image includes rectangles, ellipses, triangles, and circles, optionally combined
    with a Shepp-Logan phantom. The image can also be masked into a circular region if `tomo=True`.

    Parameters
    ----------
    npix : int, optional
        Size of the output square image (default: 512x512).
    nshapes : int, optional
        Number of each shape type to generate (default: 50).
    minsize : int, optional
        Minimum size of generated shapes (default: 5).
    maxsize : int, optional
        Maximum size of generated shapes (default: 50).
    shape : str or list of str, optional
        Specific shape(s) to generate. If None, all shape types are included (default: None).
    norm : bool, optional
        Whether to normalize the intensity of each shape individually (default: False).
    allow_overlap : bool, optional
        Whether shapes are allowed to overlap (default: True).
    tomo : bool, optional
        If True, applies a circular mask to the final image for tomography simulation (default: False).
    SL : bool, optional
        If True, adds a Shepp-Logan phantom to the image (default: False).

    Returns
    -------
    np.ndarray
        Composite phantom image with the specified characteristics.
    """
    shapes = ['rectangle', 'ellipse', 'triangle', 'circle']
    components = [phantom_random_shapes(npix, nshapes, nshapes, minsize, maxsize, s, norm, allow_overlap) for s in shapes]
    
    # Normalize each shape component separately
    components = [comp / np.max(comp) if np.max(comp) > 0 else comp for comp in components]

    # Sum all components to create the composite image
    im = np.sum(components, axis=0)

    if SL:
        im += SheppLogan(npix)
    
    im = np.clip(im, 0, 1)  # Ensure values remain in the range [0,1]

    if tomo:
        im = cirmask(im)

    return im

def sstar(npix, nstars=32):
    """
    Generate a Siemens Star phantom using xdesign.

    Parameters
    ----------
    npix : int
        Output image size (npix x npix).
    nstars : int, optional
        Number of spokes in the Siemens Star pattern (default: 32).

    Returns
    -------
    np.ndarray
        Discretized Siemens Star phantom image.
    """
    phase = SiemensStar(nstars)
    return discrete_phantom(phase, npix, prop='mass_attenuation')

def dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2):
    """
    Generate a DogaCircles phantom using xdesign.

    Parameters
    ----------
    npix : int
        Output image size (npix x npix).
    n_sizes : int, optional
        Number of different circle sizes (default: 8).
    size_ratio : float, optional
        Ratio of size reduction between consecutive circles (default: 0.75).
    n_shuffles : int, optional
        Number of random shuffles applied to the circle arrangement (default: 2).

    Returns
    -------
    np.ndarray
        Discretized DogaCircles phantom image.
    """
    phase = DogaCircles(n_sizes=n_sizes, size_ratio=size_ratio, n_shuffles=n_shuffles)
    return discrete_phantom(phase, npix, prop='mass_attenuation')

def ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05):
    """
    Generate a Slanted Squares phantom using xdesign.

    Parameters
    ----------
    npix : int
        Output image size (npix x npix).
    count : int, optional
        Number of square elements (default: 16).
    angle : float, optional
        Rotation angle of the squares in radians (default: 15 degrees).
    gap : float, optional
        Gap between squares as a fraction of their size (default: 0.05).

    Returns
    -------
    np.ndarray
        Discretized Slanted Squares phantom image.
    """
    phase = SlantedSquares(count=count, angle=angle, gap=gap)
    return discrete_phantom(phase, npix, prop='mass_attenuation')

def tri(npix, p1=(-0.3, -0.2), p2=(0.0, -0.3), p3=(0.3, -0.2)):
    """
    Generate a triangular phantom using xdesign.

    Parameters
    ----------
    npix : int
        Output image size (npix x npix).
    p1 : tuple of float, optional
        Coordinates of the first vertex (default: (-0.3, -0.2)).
    p2 : tuple of float, optional
        Coordinates of the second vertex (default: (0.0, -0.3)).
    p3 : tuple of float, optional
        Coordinates of the third vertex (default: (0.3, -0.2)).

    Returns
    -------
    np.ndarray
        Discretized triangular phantom image.
    """
    m = Mesh()
    m.append(Triangle(Point(p1), Point(p2), Point(p3)))
    phase = Phantom(geometry=m)
    phase.material = SimpleMaterial(mass_attenuation=1.0)
    return discrete_phantom(phase, npix, prop='mass_attenuation')

def face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 ):
    
    """
    Generate a stylized face phantom using xdesign.

    The face consists of a circular base with a triangular cutout (mouth) and two circular eyes.

    Parameters
    ----------
    npix : int
        Output image size (npix x npix).
    cp : tuple of float, optional
        (x, y) coordinates of the face center (default: (0.0, 0.0)).
    cr : float, optional
        Radius of the face (default: 0.5).
    tp1 : tuple of float, optional
        First vertex of the triangular cutout (default: (-0.3, -0.2)).
    tp2 : tuple of float, optional
        Second vertex of the triangular cutout (default: (0.0, -0.3)).
    tp3 : tuple of float, optional
        Third vertex of the triangular cutout (default: (0.3, -0.2)).
    e1p : tuple of float, optional
        (x, y) coordinates of the left eye center (default: (-0.2, 0.0)).
    e1r : float, optional
        Radius of the left eye (default: 0.1).
    e2p : tuple of float, optional
        (x, y) coordinates of the right eye center (default: (0.2, 0.0)).
    e2r : float, optional
        Radius of the right eye (default: 0.1).

    Returns
    -------
    np.ndarray
        Discretized face phantom image.
    """

    # Make a circle with a triangle cut out
    m = Mesh()
    m.append(Circle(Point(cp), radius=cr))
    m.append(Triangle(Point(tp1), Point(tp2), Point(tp3)))
    
    phase = Phantom(geometry=m)
    
    # Make two eyes separately
    eyeL = Phantom(geometry=Circle(Point(e1p), radius=e1r))
    eyeR = Phantom(geometry=Circle(Point(e2p), radius=e2r))
    
    phase.material = SimpleMaterial(mass_attenuation=1.0)
    eyeL.material = SimpleMaterial(mass_attenuation=1.0)
    eyeR.material = SimpleMaterial(mass_attenuation=1.0)
    
    phase.append(eyeL)
    phase.append(eyeR)
    
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    
    return(im)
    
def load_example_xanes():
    
    """
    Load a test dataset containing five XANES spectra.

    The dataset is stored in an HDF5 file and includes spectra for different materials, 
    as well as the corresponding energy values.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - sNMC: Spectrum for NMC material.
        - sNi2O3: Spectrum for Ni₂O₃.
        - sNiOH2: Spectrum for Ni(OH)₂.
        - sNiS: Spectrum for NiS.
        - sNifoil: Spectrum for Ni foil.
        - E: Energy values associated with the spectra.
    """

    fn = '%s\examples\patterns\AllSpectra.h5' %(ndtomopath())

    with h5py.File(fn, 'r') as f:
        
        print(f.keys())
        
        sNMC = np.array(f['NMC'][:])
        sNi2O3 = np.array(f['Ni2O3'][:])
        sNiOH2 = np.array(f['NiOH2'][:])
        sNiS = np.array(f['NiS'][:])
        sNifoil = np.array(f['Nifoil'][:])
    
        E = np.array(f['energy'][:])

    return(sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E)

def load_example_patterns():
    
    """
    Load a test dataset containing five diffraction patterns.

    The dataset is stored in an HDF5 file and includes diffraction patterns for different materials, 
    along with corresponding `tth` (two-theta) and `q` (momentum transfer) values.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - dpAl: Diffraction pattern for aluminum (Al).
        - dpCu: Diffraction pattern for copper (Cu).
        - dpFe: Diffraction pattern for iron (Fe).
        - dpPt: Diffraction pattern for platinum (Pt).
        - dpZn: Diffraction pattern for zinc (Zn).
        - tth: Two-theta values associated with the diffraction patterns.
        - q: Momentum transfer values associated with the diffraction patterns.
    """  
    
    fn = '%s\examples\patterns\patterns.h5' %(ndtomopath())

    with h5py.File(fn, 'r') as f:
        
        print(f.keys())
        
        dpAl = np.array(f['Al'][:])
        dpCu = np.array(f['Cu'][:])
        dpFe = np.array(f['Fe'][:])
        dpPt = np.array(f['Pt'][:])
        dpZn = np.array(f['Zn'][:])
        
        tth = np.array(f['tth'][:])
        q = np.array(f['q'][:])

    return(dpAl, dpCu, dpFe, dpPt, dpZn, tth, q)

def nDTomophantom_2D(npix, nim='One'):
    """
    Generate 2D phantom images using a combination of predefined patterns.

    This function creates 2D images using five different phantom types from xdesign: 
    - Siemens Star
    - Doga Circles
    - Slanted Squares
    - Triangle
    - Face

    The function can return either a single composite image combining all patterns 
    or multiple separate images.

    Parameters
    ----------
    npix : int
        The size of the generated images (npix x npix).
    nim : str, optional
        Specifies whether to generate a single composite image ('One') or 
        return multiple separate images as a list ('Multiple'). Default is 'One'.

    Returns
    -------
    np.ndarray or list of np.ndarray
        - If `nim='One'`: Returns a single composite image (np.ndarray).
        - If `nim='Multiple'`: Returns a list of individual images ([im1, im2, im3, im4, im5]).
    
    Raises
    ------
    ValueError
        If `nim` is not 'One' or 'Multiple'.
    """

    # Generate individual phantom images
    im1 = sstar(npix, nstars=32)
    im2 = dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2)
    im3 = ssquares(npix, count=16, angle=15 / 360 * 2 * np.pi, gap=0.05)
    im4 = tri(npix, p1=[-0.3, -0.2], p2=[0.0, -0.3], p3=[0.3, -0.2])
    im5 = face(npix, cp=[0.0, 0.0], cr=0.5, tp1=[-0.3, -0.2], tp2=[0.0, -0.3], tp3=[0.3, -0.2],
               e1p=[-0.2, 0.0], e1r=0.1, e2p=[0.2, 0.0], e2r=0.1)

    if nim == 'One':
        im = im1 + im2 + im3 + im4 + im5
        im /= np.max(im)  # Normalize image to [0,1]
        return im

    if nim == 'Multiple':
        return [im1, im2, im3, im4, im5]

    raise ValueError("Invalid value for `nim`. Expected 'One' or 'Multiple'.")

def nDTomophantom_3D(npix, use_spectra = False, spectra = None, nz = 100, imgs = None, 
                     indices = 'Random', inds = None,  norm = 'Volume'):
    
    """
    Generate a 3D phantom dataset using a list of component images.

    This function creates either a 3D spatial volume dataset or a 2D chemical dataset 
    (spatial + spectral dimension). The dataset is generated using a list of predefined 
    component images or user-provided images. The function allows customization of 
    how images are assigned across the third dimension (Z or spectral channels).

    Parameters
    ----------
    npix : int
    Number of pixels for each generated image (resulting in a square image of shape `npix x npix`).
    use_spectra : bool, optional
        If `True`, generates a 2D + spectral dataset.
        If `False` (default), creates a purely spatial 3D volume.
    spectra : list of np.ndarray, optional
        A list of component spectra to be assigned to the dataset.
        If `None` (default), example diffraction patterns are loaded.
    nz : int, optional
        Number of slices in the volume (Z-dimension) when `use_spectra=False`.
        Ignored when `use_spectra=True`.
    imgs : list of np.ndarray, optional
        A list of images to be used as the basis for constructing the dataset.
        If `None`, default component images are generated using `nDphantom_2D()`.
    indices : str, optional
        Defines how component images are assigned along the Z-dimension. Options:
        - 'Random' (default): Randomly assigns image slices.
        - 'All': Uses all images at all Z positions.
        - 'Custom': Assigns images based on provided index list (`inds`).
    inds : list of lists, optional
        A list containing two sublists: `[inds_in, inds_fi]`, specifying the 
        starting and ending indices where each component image will appear.
        Required only if `indices='Custom'`.
    norm : str, optional
        Defines how normalization is applied to the dataset. Options:
        - 'Volume' (default): Normalizes the entire 3D dataset with respect to the 
        highest intensity.
        - 'Images': Normalizes each image separately across the third dimension.

    Returns
    -------
    np.ndarray
    A 3D phantom volume with dimensions `(npix, npix, nz/nch)`, where `nz` is the 
    number of Z-slices or `nch` is the number of spectral channels.

    Raises
    ------
    ValueError
    If `indices='Custom'` is selected but `inds` is not provided.

    Notes
    -----
    - If `use_spectra=True`, the function returns a 2D chemical dataset where 
    each spectral component is multiplied by a spatial mask.
    - If `use_spectra=False`, a purely spatial 3D volume is created with randomized or 
    custom placement of images.

    Examples
    --------
    To create a 2D chemical dataset:
    vol = nDTomophantom_3D(npix=256, use_spectra='Yes', indices='All', norm='No')

    To create a 3D volume dataset with random placement:
    vol = nDTomophantom_3D(npix=256, nz=100, indices='Random', norm='Volume')
    """
    
    if imgs is None:
        
        im1, im2, im3, im4, im5 = nDTomophantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]
    
    if not use_spectra:

        # Create a volume with 3 spatial dimensions        

        vol =  np.zeros((npix, npix, nz))
        
        if indices == 'Random':
            
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.zeros((len(imgs)))
        
            for ii in range(len(imgs)):
                
                inds_in[ii] = np.random.randint(0, nz-2)
                inds_fi[ii] = np.random.randint(inds_in[ii]+1, nz)
                
        elif indices == 'All':
                    
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.ones((len(imgs)))*vol.shape[2]  
            
        elif indices == 'Custom':
            
            inds_in, inds_fi = inds
            
        for ii in tqdm(range(len(imgs))):
            
            vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] = (vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] + 
                                                              np.transpose(np.tile(imgs[ii], (len(np.arange(inds_in[ii],inds_fi[ii])), 1, 1)), (2,1,0)))
    else:                
        # Create a volume with 2 spatial dimensions and 1 spectral dimension

        if spectra is None:
            sp1, sp2, sp3, sp4, sp5, tth, q = load_example_patterns()
            spectra = [sp1, sp2, sp3, sp4, sp5]
        else:
            sp1 = spectra[0]

        nch = len(sp1)
        vol =  np.zeros((npix, npix, nch))
        
        if indices == 'Random':
            
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.zeros((len(imgs)))
        
            for ii in range(len(imgs)):
                
                inds_in[ii] = np.random.randint(0, nch-2)
                inds_fi[ii] = np.random.randint(inds_in[ii]+1, nch)
                
        elif indices == 'All':
                    
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.ones((len(imgs)))*vol.shape[2]  
            
        elif indices == 'Custom':
            
            inds_in, inds_fi = inds
        
        for ii in tqdm(range(len(imgs))):
            
            vol_tmp = np.tile(spectra[ii], (npix, npix, 1))*np.transpose(np.tile(imgs[ii], (nch, 1, 1)), (2,1,0))
            
            vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] = vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] + vol_tmp[:,:,int(inds_in[ii]):int(inds_fi[ii])]     
            
    if norm == 'Volume':
        
        vol = vol/np.max(vol)
        
    elif norm == 'Images':
        
        for ii in tqdm(range(vol.shape[2])):
        
            vol[:,:,ii] = vol[:,:,ii]/np.max(vol[:,:,ii])          
        
    return(vol)
    

def nDTomophantom_4D(npix, nzt, vtype = 'Spectral', imgs = None, indices = 'Random', inds = None, spectra = None,  norm = 'Volume'):
    
    """
    Generate a 4D phantom dataset using a list of component images.

    This function creates either:
    - A 3D spatial + 1D spectral dataset (`vtype='Spectral'`).
    - A 3D spatial + 1D temporal dataset (`vtype='Temporal'`).

    The dataset is constructed using either predefined component images or user-provided images.
    Users can control how images are assigned across the third and fourth dimensions.

    Parameters
    ----------
    npix : int
        Number of pixels for each generated image (resulting in a square image of shape `(npix, npix)`).
    nzt : int
        Number of slices in the fourth dimension (temporal points if `vtype='Temporal'`, 
        or spectral channels if `vtype='Spectral'`).
    vtype : str, optional
        Defines the type of 4D dataset to generate. Options:
        - 'Spectral' (default): Creates a 3D spatial + 1D spectral dataset.
        - 'Temporal': Creates a 3D spatial + 1D temporal dataset.
    imgs : list of np.ndarray, optional
        A list of images used as spatial components.
        If `None`, default component images are generated using `nDTomophantom_2D()`.
    indices : str, optional
        Defines how component images are assigned along the fourth dimension. Options:
        - 'Random' (default): Randomly assigns image slices.
        - 'All': Uses all images at all positions in the fourth dimension.
        - 'Custom': Assigns images based on a provided index list (`inds`).
    inds : list of lists, optional
        A list containing two sublists: `[inds_in, inds_fi]`, specifying the 
        starting and ending indices where each component image will appear.
        Required only if `indices='Custom'`.
    spectra : list of np.ndarray, optional
        A list of component spectra used when `vtype='Spectral'`.
        If `None`, example diffraction patterns are loaded.
    norm : str, optional
        Defines how normalization is applied to the dataset. Options:
        - 'Volume' (default): Normalizes the entire 4D dataset with respect to the 
        highest intensity.

    Returns
    -------
    np.ndarray
    - Shape `(npix, npix, nzt, nch)` if `vtype='Spectral'` (3D spatial + 1D spectral).
    - Shape `(npix, npix, nch, nzt)` if `vtype='Temporal'` (3D spatial + 1D temporal).

    Raises
    ------
    ValueError
    If `indices='Custom'` is selected but `inds` is not provided.

    Notes
    -----
    - If `vtype='Spectral'`, the function returns a 3D spatial dataset with a spectral dimension.
    - If `vtype='Temporal'`, the function returns a 3D spatial dataset with a temporal evolution 
    where spectral components change over time.

    Examples
    --------
    To create a 4D spectral dataset:
    vol4D = nDphantom_4D(npix=256, nzt=100, vtype='Spectral', indices='All')

    To create a 4D temporal dataset with dynamic spectra:
    vol4D = nDphantom_4D(npix=256, nzt=50, vtype='Temporal', indices='Random')
    """


    if imgs is None:
        im1, im2, im3, im4, im5 = nDTomophantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]
    
    if spectra is None:
        sp1, sp2, sp3, sp4, sp5, tth, q = load_example_patterns()
        spectra = [sp1, sp2, sp3, sp4, sp5]
        nch = len(sp1)
    else:
        sp1, sp2, sp3, sp4, sp5 = spectra    
        nch = len(sp1)
        
    
    if vtype == 'Temporal':
        
        vol4D = np.zeros((npix, npix, nch, nzt))
        
        if indices == 'Random':
            
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.zeros((len(imgs)))
        
            for ii in range(len(imgs)):
                
                inds_in[ii] = np.random.randint(0, nch-2)
                inds_fi[ii] = np.random.randint(inds_in[ii]+1, nch)
                
        elif indices == 'All':
                    
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.ones((len(imgs)))*nch       
            
        elif indices == 'Custom':
            
            inds_in, inds_fi = inds           
        
        xold = np.arange(0, nch)
        tstep = np.ceil(nch*0.01*3)
        
        # Now we have tp define the behaviour of each component (-1, 0, 1) which specifies the direction that the component is moving (0 means no movement)
        
        compbeh = np.zeros((len(imgs)))
        for ii in range(len(compbeh)):
            compbeh[ii] = np.random.randint(-1,2)
        
        for ii in tqdm(range(nzt)):
            
            spnewlist = []
            
            for jj in range(len(spectra)):
            
                f = interp1d(xold, spectra[jj], kind='linear', bounds_error=False, fill_value=0)
                
                spnewlist.append(f(xold + compbeh[jj]*ii*tstep))
        
            vol4D[:,:,:,ii] = nDTomophantom_3D(npix=npix, use_spectra = 'Yes', spectra = spnewlist, nz=nch, imgs=imgs, indices = 'Custom', inds = [inds_in, inds_fi], norm = 'No')

    elif vtype == 'Spectral':
        
        vol4D = np.zeros((npix, npix, nzt, nch))
        
        if indices == 'Random':
            
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.zeros((len(imgs)))
        
            for ii in range(len(imgs)):
                
                inds_in[ii] = np.random.randint(0, nzt-2)
                inds_fi[ii] = np.random.randint(inds_in[ii]+1, nzt)
                
        elif indices == 'All':
                    
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.ones((len(imgs)))*nzt       
            
        elif indices == 'Custom':
            
            inds_in, inds_fi = inds            
            
        # In order to avoid storing in memory multiple 4D matrices, we go per z position and create a 3D dataset (i.e. 2D spatial, 1D spectral) for each component        
        
        for ii in tqdm(range(nzt)):
            
            vol3D_tmp = np.zeros((npix, npix, nch))
            
            for jj in range(len(imgs)):
            
                if inds_in[jj]<=ii<=inds_fi[jj]:
                    
                    vol3D_tmp = vol3D_tmp + np.tile(spectra[jj], (npix, npix, 1))*np.transpose(np.tile(imgs[jj], (nch, 1, 1)), (2,1,0))
            
            vol4D[:,:,ii,:] = vol4D[:,:,ii,:] + vol3D_tmp
    
    if norm == 'Volume':
        
        vol4D = vol4D/np.max(vol4D)
        
    
    return(vol4D)

def nDphantom_5D(npix, nz, nt, imgs = None, indices = 'Random', spectra = None):

    """
    Generate a 5D phantom dataset using a list of component images.

    This function creates a 5D dataset with three spatial dimensions, 
    one spectral dimension, and one temporal dimension. The dataset 
    is constructed using either predefined component images or user-provided images. 
    The spectral dimension evolves dynamically over time, simulating 
    temporal changes in spectral characteristics.

    Parameters
    ----------
    npix : int
        Number of pixels for each generated image (resulting in a square image of shape `(npix, npix)`).
    nz : int
        Number of slices in the Z-dimension (depth of the volume).
    nt : int
        Number of temporal points (time frames).
    imgs : list of np.ndarray, optional
        A list of images used as spatial components.
        If `None`, default component images are generated using `nDTomophantom_2D()`.
    indices : str, optional
        Defines how component images are assigned along the Z-dimension. Options:
        - 'Random' (default): Randomly assigns image slices.
        - 'All': Uses all images at all Z positions.
    spectra : list of np.ndarray, optional
        A list of component spectra used to define the spectral evolution.
        If `None`, example diffraction patterns are loaded.

    Returns
    -------
    np.ndarray
        A 5D phantom dataset with dimensions `(npix, npix, nz, nch, nt)`, where:
        - `nz` is the number of slices in the Z-dimension.
        - `nch` is the number of spectral channels.
        - `nt` is the number of time frames.

    Notes
    -----
    - The spectral dimension changes dynamically over time, where each spectral 
      component is interpolated based on a random behavior assigned to each component.
    - This function relies on `nDTomophantom_4D()` to generate each time frame.
    - The behavior of spectral components is controlled by a random movement 
      parameter (`-1`, `0`, or `1`), indicating whether the component shifts 
      down, remains static, or shifts up over time.

    Examples
    --------
    To create a 5D dataset with evolving spectral characteristics over time:
        vol5D = nDphantom_5D(npix=256, nz=50, nt=10, indices='All')

    To create a 5D dataset with randomly assigned component images in Z:
        vol5D = nDphantom_5D(npix=256, nz=50, nt=10, indices='Random')
    """
    
    if imgs is None:
        im1, im2, im3, im4, im5 = nDTomophantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]
    
    if spectra is None:
        sp1, sp2, sp3, sp4, sp5, tth, q = load_example_patterns()
        spectra = [sp1, sp2, sp3, sp4, sp5]
        nch = len(sp1)
    else:
        sp1, sp2, sp3, sp4, sp5 = spectra    
        nch = len(sp1)
        
    
    vol5D = np.zeros((npix, npix, nz, nch, nt))    

    # Now we have tp define the behaviour of each component (-1, 0, 1) which specifies the direction that the component is moving (0 means no movement)
    xold = np.arange(0, nch)
    tstep = np.ceil(nch*0.01*3)
    compbeh = np.zeros((len(imgs)))
    for ii in range(len(compbeh)):
        compbeh[ii] = np.random.randint(-1,2)
        
    for ii in tqdm(range(vol5D.shape[4])):
        
        spnewlist = []
        
        for jj in range(len(spectra)):
        
            f = interp1d(xold, spectra[jj], kind='linear', bounds_error=False, fill_value=0)
            
            spnewlist.append(f(xold + compbeh[jj]*ii*tstep))        
        
        vol5D[:,:,:,:,ii] = nDTomophantom_4D(npix=npix, nzt=nz, vtype = 'Spectral', imgs=imgs, indices = 'All',  spectra = spnewlist,  norm = 'No')


    return(vol5D)


def nDphantom_2Dmap(vol, dim = 0):
    
    """
    Generate a projection map from a 3D volume dataset.

    This function computes a 2D projection of a 3D volume by summing over 
    a specified axis. If the input is a 4D dataset, the result is a 3D volume 
    projected along the specified dimension.

    Parameters
    ----------
    vol : np.ndarray
        A 3D or 4D array representing the volume dataset.
        - If 3D (`shape=(X, Y, Z)`), the output is a 2D image (`shape=(X, Y)`, etc.).
        - If 4D (`shape=(X, Y, Z, C)`), the output is a 3D volume (`shape=(X, Y, C)`, etc.).
    dim : int, optional
        The axis along which the projection is computed. Default is `0`.

    Returns
    -------
    np.ndarray
        A 2D or 3D array containing the projected sum along the selected dimension.

    Notes
    -----
    - This function performs a simple sum projection (`sum()` along `dim`).
    - For a max-intensity projection, consider using `np.max(vol, axis=dim)`.
    - The function applies `np.squeeze()` to remove singleton dimensions.

    Examples
    --------
    To create a 2D projection from a 3D volume along the Z-axis:
        proj = nDphantom_2Dmap(vol, dim=2)

    To create a 3D projection from a 4D dataset along the spectral axis:
        proj = nDphantom_2Dmap(vol, dim=3)
    """

    map2D = np.squeeze(np.sum(vol, axis=dim))

    return(map2D)


































