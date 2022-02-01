# -*- coding: utf-8 -*-
"""
Create phantoms (2D/3D) for image processing and analysis experiments

@author: Antony Vamvakeros
"""

from nDTomo.utils.misc import ndtomopath
from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares
import numpy as np
import h5py
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from skimage.draw import random_shapes
from scipy.interpolate import interp1d

# Might need to convert the functions to class methods

def SheppLogan(npix):

    '''
    Create a Shepp Logan phantom using skimage
    '''

    im = shepp_logan_phantom()
    im = rescale(im, scale=npix/im.shape[0], mode='reflect')
    return(im)

def sstar(npix, nstars=32):
    
    '''
    SiemensStar image from xdesign
    '''
    
    phase = SiemensStar(nstars)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2):

    '''
    DogaCircles image from xdesign
    '''

    phase = DogaCircles(n_sizes=n_sizes, size_ratio=size_ratio, n_shuffles=n_shuffles)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05):

    '''
    SlantedSquares image from xdesign
    '''

    phase = SlantedSquares(count=count, angle=angle, gap=gap)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2]):
    
    '''
    Triangle image from xdesign
    '''

    m = Mesh()
    m.append(Triangle(Point(p1), Point(p2), Point(p3)))
    phase = Phantom(geometry=m)
    phase.material = SimpleMaterial(mass_attenuation=1.0)
    im = discrete_phantom(phase, npix, prop='mass_attenuation')
    return(im)

def face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 ):
    
    '''
    Face image from xdesign
    '''
    
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
    

def load_example_patterns():
    
    '''
    Load a test dataset containing five diffraction patterns
    '''    
    
    
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

def nDphantom_2D(npix, nim = 'One'):
        
    '''
    Create phantom image(s) using a combination of five images created with xdesign: SlantedSquares, SiemensStar, DogaCircles, Triangle and Face
    Inputs:
        npix: number of pixels for the generated image(s); it generates squared image(s)
        nim: string corresponding to number of images, can be 'One' or 'Multiple'
    '''    
    
    im1 = sstar(npix, nstars=32)
    im2 = dcircles(npix, n_sizes=8, size_ratio=0.75, n_shuffles=2)
    im3 = ssquares(npix, count=16, angle=15/360*2*np.pi, gap=0.05)
    im4 = tri(npix, p1 = [-0.3, -0.2], p2 = [0.0, -0.3], p3 = [0.3, -0.2])
    im5 = face(npix, cp = [0.0, 0.0], cr=0.5, tp1 = [-0.3, -0.2], tp2 = [0.0, -0.3], tp3 = [0.3, -0.2],
         e1p = [-0.2, 0.0], e1r=0.1, e2p = [0.2, 0.0], e2r=0.1 )
    
    if nim == 'One':
    
        im = im1 + im2 + im3 + im4 + im5
        im = im/np.max(im)
    
    elif nim == 'Multiple':
        
        im = [im1, im2, im3, im4, im5]
        
    else:
        
        print('Wrong input for nim')
        
    return(im)

def nDphantom_3D(npix, use_spectra = 'No', spectra = None, nz = 100, imgs = None, indices = 'Random', inds = None,  norm = 'Volume'):
    
    '''
    Create a 3D phantom dataset using a list of component images
    The user can provide a list of the images
    Inputs:
        npix: number of pixels for the generated image(s) comprising the volume dataset; it generates squared image(s)
        use_spectra: string ('Yes'/'No') for using a list of spectra to create the volume dataset
        spectra: list of component spectra; if not provided, it will use 5 example patterns provide at nDTomo
        nz: number of images comprising the volume dataset; if spectra are used then nz is not taken into account and does not need to be provided
        imgs: list of images; if None (default), it will use the nDphantom_2D to create 5 component images
        indices: string (options are 'Random', 'All' and 'Custom'); specifies if the component images will be used in all z positions in the volume dataset ('All') or not ('Random'/ 'Custom')
        inds: a list containing the indices for the z positions where each component will appear, the list contains two sublists: inds = [inds_in, inds_fi]
        norm: string (options are 'Volume', 'Image') for normalising the data; 'Volume' normalises the whole 3D image stack with respect to the highest intensity, 
            'Image' normalises each image in the volume dataset
    Output:
        vol: volume with dimensions (npix, npix, nz/nch)
        
    For example, to create a 2D chemical dataset:
        vol = nDphantom_3D(npix, use_spectra = 'Yes', indices = 'All',  norm = 'No')
    to create a 3D volume dataset:
        vol = nDphantom_3D(npix, nz = 100, indices = 'Random', norm = 'Volume')
    '''    
    
    if imgs is None:
        
        im1, im2, im3, im4, im5 = nDphantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]
    
    if use_spectra == 'No':

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
            
        for ii in range(len(imgs)):
            
            vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] = (vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] + 
                                                              np.transpose(np.tile(imgs[ii], (len(np.arange(inds_in[ii],inds_fi[ii])), 1, 1)), (2,1,0)))
    elif use_spectra == 'Yes':
                
        if spectra is None:
            sp1, sp2, sp3, sp4, sp5, tth, q = load_example_patterns()
            spectra = [sp1, sp2, sp3, sp4, sp5]
        else:
            sp1, sp2, sp3, sp4, sp5 = spectra

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
        
        for ii in range(len(imgs)):
            
            vol_tmp = np.tile(spectra[ii], (npix, npix, 1))*np.transpose(np.tile(imgs[ii], (nch, 1, 1)), (2,1,0))
            
            vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] = vol[:,:,int(inds_in[ii]):int(inds_fi[ii])] + vol_tmp[:,:,int(inds_in[ii]):int(inds_fi[ii])]     
            
    if norm == 'Volume':
        
        vol = vol/np.max(vol)
        
    elif norm == 'Images':
        
        for ii in range(vol.shape[2:]):
        
            vol[:,:,ii] = vol[:,:,ii]/np.max(vol[:,:,ii])          
        
    return(vol)
    

def nDphantom_4D(npix, nz, imgs = None, indices = 'Random', vtype = 'Spectral', spectra = None):
    
    '''
    Create a 4D phantom dataset using a list of component images
    The user can provide a list of the images
    Inputs:    
        npix: number of pixels for the generated image(s) comprising the volume dataset; it generates squared image(s)
        nz: number of images comprising the volume dataset
        imgs: list of images; if None (default), it will use the nDphantom_2D to create 5 component images
        indices: string (options are 'Random', 'All'); specifies if the component images will be used in all z positions in the volume dataset ('All') or not ('Random')        
        vtype: string ('Spectral'/'Temporal') for the type of the 4D matrix; 'Spectral' correponds to 3D spatial + 1D spectral, 'Temporal' correponds to 3D spatial + 1D temporal
        spectra: list of component spectra; if not provided, it will use 5 example patterns provide at nDTomo
    Output:
        vol4D: volume with dimensions (npix, npix, nz, nch)    
    '''

    if imgs is None:
        im1, im2, im3, im4, im5 = nDphantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]
    
    if spectra is None:
        sp1, sp2, sp3, sp4, sp5, tth, q = load_example_patterns()
        spectra = [sp1, sp2, sp3, sp4, sp5]
        nch = len(sp1)
    else:
        sp1, sp2, sp3, sp4, sp5 = spectra    
        nch = len(sp1)
        
    
    vol4D = np.zeros((npix, npix, nz, nch))
    
    
    if vtype == 'Temporal':
        
        xold = np.arange(0, nch)
        tstep = np.ceil(nch*0.01)
        
        for ii in range(nch):
            
            spnewlist = []
            
            for jj in range(len(spectra)):
            
                f = interp1d(xold, spectra[jj], kind='linear', bounds_error=False, fill_value=0)
                
                spnewlist.append(f(xold + jj*tstep))
        
            vol4D[:,:,:,ii] = nDphantom_3D(npix, nch, imgs, indices = 'All', use_spectra = 'Yes', spectra = spnewlist, norm = 'No')
    
    elif vtype == 'Spectral':
        
        if indices == 'Random':
            
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.zeros((len(imgs)))
        
            for ii in range(len(imgs)):
                
                inds_in[ii] = np.random.randint(0, nz-2)
                inds_fi[ii] = np.random.randint(inds_in[ii]+1, nz)
                
        elif indices == 'All':
                    
            inds_in = np.zeros((len(imgs)))
            inds_fi = np.ones((len(imgs)))*nz       
            
        # In order to avoid storing in memory multiple 4D matrices, we go per z position and create a 3D dataset (i.e. 2D spatial, 1D spectral) for each component        
        
        for ii in range(nz):
            
            vol3D_tmp = np.zeros((npix, npix, nch))
            
            for jj in range(len(imgs)):
            
                if inds_in[jj]<=ii<=inds_fi[jj]:
                    
                    vol3D_tmp = vol3D_tmp + np.tile(spectra[jj], (npix, npix, 1))*np.transpose(np.tile(imgs[jj], (nz, 1, 1)), (2,1,0))
            
            vol4D[:,:,ii,:] = vol4D[:,:,ii,:] + vol3D_tmp
    
    vol4D = vol4D/np.max(vol4D)
    
    return(vol4D)

# def xrdmap(npix, nz=10, imgs = None, dps = None):
    
#     '''
#     Calculate an XRD map from a simulated 3D-XRD-CT dataset
#     '''
    
#     xrdct3d = phantom_3Dxrdct(npix, nz, imgs, dps)
    
#     xrdmap = np.squeeze(np.sum(xrdct3d, axis=1))
    
#     return(xrdmap)

# def phantom5c_3Dxrdct(npix, nz = 5, imgs = None, dps = None):
    
#     '''
#     3D-XRD-CT phantom using 5 components
#     User can provide a list of images and a list of diffraction patterns
#     '''
    
#     if imgs is None:
#         imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
#     else:
#         imAl, imCu, imFe, imPt, imZn = imgs
    
#     if dps is None:
#         dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
#     else:
#         dpAl, dpCu, dpFe, dpPt, dpZn = dps
    
#     vol_Al = np.tile(dpAl, (npix, npix, 1))
#     vol_Cu = np.tile(dpCu, (npix, npix, 1))
#     vol_Fe = np.tile(dpFe, (npix, npix, 1))
#     vol_Pt = np.tile(dpPt, (npix, npix, 1))
#     vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
#     xrdct =  np.zeros((npix, npix, len(dpAl)))
    
#     for ii in range(xrdct.shape[2]):
        
#         xrdct[:,:,ii] = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
#               + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
    
#     xrdct = np.reshape(xrdct, (npix, npix, 1, xrdct.shape[2]))
    
#     xrdct3d = np.zeros_like(xrdct)
    
#     for ii in range(nz):
        
#         xrdct3d = np.concatenate((xrdct3d, xrdct), axis = 2)
    
#     xrdct3d = xrdct3d[:,:,1:,:]
    
#     return(xrdct3d)


# def load_example_xanes():
    
#     '''
#     Load a test dataset containing five XANES spectra
#     '''    

#     fn = '%s\examples\patterns\AllSpectra.h5' %(ndtomopath())

#     with h5py.File(fn, 'r') as f:
        
#         print(f.keys())
        
#         sNMC = np.array(f['NMC'][:])
#         sNi2O3 = np.array(f['Ni2O3'][:])
#         sNiOH2 = np.array(f['NiOH2'][:])
#         sNiS = np.array(f['NiS'][:])
#         sNifoil = np.array(f['Nifoil'][:])
    
#         E = np.array(f['energy'][:])

#     return(sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E)

# def phantom5c_xanesct(npix, imgs = None, spectra = None):
    
#     '''
#     XANES-CT phantom using 5 components
#     User can provide a list of images and a list of spectra
#     '''
    
#     if imgs is None:
#         imNMC, imNi2O3, imNiOH2, imNiS, imNifoil = phantom5c(npix)
#     else:
#         imNMC, imNi2O3, imNiOH2, imNiS, imNifoil = imgs
    
#     if spectra is None:
#         sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E = load_example_xanes()
#     else:
#         sNMC, sNi2O3, sNiOH2, sNiS, sNifoil = spectra    
        
#     vol_NMC = np.tile(sNMC, (npix, npix, 1))
#     vol_Ni2O3 = np.tile(sNi2O3, (npix, npix, 1))
#     vol_NiOH2 = np.tile(sNiOH2, (npix, npix, 1))
#     vol_NiS = np.tile(sNiS, (npix, npix, 1))
#     vol_Nifoil = np.tile(sNifoil, (npix, npix, 1))
    
#     xanesct =  np.zeros_like(vol_NMC)
    
#     for ii in range(xanesct.shape[2]):
        
#         xanesct[:,:,ii] = (vol_NMC[:,:,ii]*imNMC +  vol_Ni2O3[:,:,ii]*imNi2O3 + vol_NiOH2[:,:,ii]*imNiOH2
#               + vol_NiS[:,:,ii]*imNiS + vol_Nifoil[:,:,ii]*imNifoil)
        
#     return(xanesct)



def phantom_random_shapes(sz=368, min_shapes=3, max_shapes=10, min_size=5, max_size=50, norm=False):

    '''
    Create an image with random shapes
    '''    

    im, _ = random_shapes((sz, sz), min_shapes=min_shapes, max_shapes=max_shapes, multichannel=False,
                             min_size=min_size, max_size=max_size, allow_overlap=True)

    im = np.where(im==255, 1, im)
    
    if norm == True:
        im = im/np.max(im)
    
    return(im)




































