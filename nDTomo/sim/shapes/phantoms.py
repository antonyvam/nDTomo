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

def nDphantom_microct(npix, nz = 100, imgs = None):
    
    '''
    Create a 3D phantom using a list of component images
    The user can provide a list of the images
    Inputs:
        npix: number of pixels for the generated image(s) comprising the volume stack; it generates squared image(s)
        nz: number of images comprising the volume stack
        imgs: list of images; if None (default), it will use the nDphantom_2D to create 5 component images
    Output:
        vol: volume with dimensions (npix, npix, nz)
    '''    
    
    if imgs is None:
        im1, im2, im3, im4, im5 = nDphantom_2D(npix, 'Multiple')
        imgs = [im1, im2, im3, im4, im5]

    microct =  np.zeros((npix, npix, nz))
    
    inds_in = np.zeros((len(imgs)))
    inds_fi = np.zeros((len(imgs)))
    
    for ii in range(len(imgs)):
        
        inds_in[ii] = np.random.randint(0, nz-2)
        inds_fi[ii] = np.random.randint(inds_in[ii]+1, nz)
   
    
    for ii in range(len(imgs)):
        
        microct[:,:,int(inds_in[ii]):int(inds_fi[ii])] = (microct[:,:,int(inds_in[ii]):int(inds_fi[ii])] + 
                                                          np.transpose(np.tile(imgs[ii], (len(np.arange(inds_in[ii],inds_fi[ii])), 1, 1)), (2,1,0)))
        
    return(microct)
    

def phantom5c_microct(npix, imgs = None, dps = None):
    
    '''
    micro-CT phantom using 5 components
    User can provide a list of 5 images and a list of 5 spectra
    '''
    
    if imgs is None:
        imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
    else:
        imAl, imCu, imFe, imPt, imZn = imgs
    
    if dps is None:
        dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
    else:
        dpAl, dpCu, dpFe, dpPt, dpZn = dps
    
    vol_Al = np.tile(dpAl, (npix, npix, 1))
    vol_Cu = np.tile(dpCu, (npix, npix, 1))
    vol_Fe = np.tile(dpFe, (npix, npix, 1))
    vol_Pt = np.tile(dpPt, (npix, npix, 1))
    vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
    microct =  np.zeros_like(vol_Al)
    
    for ii in range(vol_Al.shape[2]):
        
        im = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
              + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
        
        microct[:,:,ii] = im/np.max(im)    

    return(microct)

def phantom5c_xrdct(npix, imgs = None, dps = None):
    
    '''
    XRD-CT phantom using 5 components
    User can provide a list of images and a list of diffraction patterns
    '''
    
    if imgs is None:
        imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
    else:
        imAl, imCu, imFe, imPt, imZn = imgs
    
    if dps is None:
        dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
    else:
        dpAl, dpCu, dpFe, dpPt, dpZn = dps
    
    vol_Al = np.tile(dpAl, (npix, npix, 1))
    vol_Cu = np.tile(dpCu, (npix, npix, 1))
    vol_Fe = np.tile(dpFe, (npix, npix, 1))
    vol_Pt = np.tile(dpPt, (npix, npix, 1))
    vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
    xrdct =  np.zeros_like(vol_Al)
    
    for ii in range(xrdct.shape[2]):
        
        xrdct[:,:,ii] = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
              + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
        
    return(xrdct)

def phantom_3Dxrdct(npix, nz = 100, imgs = None, dps = None):
    
    '''
    micro-CT phantom using component images
    User can provide a list of the images
    '''    
    
    if imgs is None:
        im1, im2, im3, im4, im5 = phantom5c(npix)
        imgs = [im1, im2, im3, im4, im5]

    if dps is None:
        dp1, dp2, dp3, dp4, dp5, tth, q = load_example_patterns()
        dps = dp1, dp2, dp3, dp4, dp5
               
    nch = len(dp1)
    inds_in = np.zeros((len(imgs)))
    inds_fi = np.zeros((len(imgs)))
    
    for ii in range(len(imgs)):
        
        inds_in[ii] = np.random.randint(0, nz-2)
        inds_fi[ii] = np.random.randint(inds_in[ii]+1, nz)
   
    xrdct3d = np.zeros((npix, npix, nz, nch))
    
    vols = []
    for ii in range(len(imgs)):
        vols.append(np.tile(dps[ii], (len(np.arange(inds_in[ii],inds_fi[ii])), npix, npix, 1)).transpose(1,2,0,3))
        
    for ii in range(len(vols)):
                
        for jj in range(vols[ii].shape[2]):
            
            for kk in range(nch):
        
                vols[ii][:,:,jj,kk] = vols[ii][:,:,jj,kk] * imgs[ii]
        
    for ii in range(len(imgs)):
        
        xrdct3d_tmp = np.zeros((npix, npix, nz, nch))
        
        xrdct3d_tmp[:,:,int(inds_in[ii]):int(inds_fi[ii]),:] = vols[ii]
        
        xrdct3d = xrdct3d + xrdct3d_tmp
        
    return(xrdct3d)

def xrdmap(npix, nz=10, imgs = None, dps = None):
    
    '''
    Calculate an XRD map from a simulated 3D-XRD-CT dataset
    '''
    
    xrdct3d = phantom_3Dxrdct(npix, nz, imgs, dps)
    
    xrdmap = np.squeeze(np.sum(xrdct3d, axis=1))
    
    return(xrdmap)

def phantom5c_3Dxrdct(npix, nz = 5, imgs = None, dps = None):
    
    '''
    3D-XRD-CT phantom using 5 components
    User can provide a list of images and a list of diffraction patterns
    '''
    
    if imgs is None:
        imAl, imCu, imFe, imPt, imZn = phantom5c(npix)
    else:
        imAl, imCu, imFe, imPt, imZn = imgs
    
    if dps is None:
        dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
    else:
        dpAl, dpCu, dpFe, dpPt, dpZn = dps
    
    vol_Al = np.tile(dpAl, (npix, npix, 1))
    vol_Cu = np.tile(dpCu, (npix, npix, 1))
    vol_Fe = np.tile(dpFe, (npix, npix, 1))
    vol_Pt = np.tile(dpPt, (npix, npix, 1))
    vol_Zn = np.tile(dpZn, (npix, npix, 1))
    
    xrdct =  np.zeros((npix, npix, len(dpAl)))
    
    for ii in range(xrdct.shape[2]):
        
        xrdct[:,:,ii] = (vol_Al[:,:,ii]*imAl +  vol_Cu[:,:,ii]*imCu + vol_Fe[:,:,ii]*imFe
              + vol_Pt[:,:,ii]*imPt + vol_Zn[:,:,ii]*imZn)
    
    xrdct = np.reshape(xrdct, (npix, npix, 1, xrdct.shape[2]))
    
    xrdct3d = np.zeros_like(xrdct)
    
    for ii in range(nz):
        
        xrdct3d = np.concatenate((xrdct3d, xrdct), axis = 2)
    
    xrdct3d = xrdct3d[:,:,1:,:]
    
    return(xrdct3d)


def load_example_xanes():
    
    '''
    Load a test dataset containing five XANES spectra
    '''    

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

def phantom5c_xanesct(npix, imgs = None, spectra = None):
    
    '''
    XANES-CT phantom using 5 components
    User can provide a list of images and a list of spectra
    '''
    
    if imgs is None:
        imNMC, imNi2O3, imNiOH2, imNiS, imNifoil = phantom5c(npix)
    else:
        imNMC, imNi2O3, imNiOH2, imNiS, imNifoil = imgs
    
    if spectra is None:
        sNMC, sNi2O3, sNiOH2, sNiS, sNifoil, E = load_example_xanes()
    else:
        sNMC, sNi2O3, sNiOH2, sNiS, sNifoil = spectra    
        
    vol_NMC = np.tile(sNMC, (npix, npix, 1))
    vol_Ni2O3 = np.tile(sNi2O3, (npix, npix, 1))
    vol_NiOH2 = np.tile(sNiOH2, (npix, npix, 1))
    vol_NiS = np.tile(sNiS, (npix, npix, 1))
    vol_Nifoil = np.tile(sNifoil, (npix, npix, 1))
    
    xanesct =  np.zeros_like(vol_NMC)
    
    for ii in range(xanesct.shape[2]):
        
        xanesct[:,:,ii] = (vol_NMC[:,:,ii]*imNMC +  vol_Ni2O3[:,:,ii]*imNi2O3 + vol_NiOH2[:,:,ii]*imNiOH2
              + vol_NiS[:,:,ii]*imNiS + vol_Nifoil[:,:,ii]*imNifoil)
        
    return(xanesct)



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




































