# -*- coding: utf-8 -*-
"""
Tensorflow functions for tomography
"""

import tensorflow as tf
import tensorflow_addons as tfa
import math as m
import numpy as np

def tf_tomo_transf(im):
    return(tf.transpose(tf.reshape(im, (im.shape[0], im.shape[1], 1, 1)), (2, 0, 1, 3)))

def tf_tomo_squeeze(im):
    return(im[0,:,:,0])

def tf_create_angles(nproj, scan = '180'):
    
    '''
    Create the projection angles
    '''
    
    if scan=='180':
        theta = np.arange(0, 180, 180/nproj)
    elif scan=='360':
        theta = np.arange(0, 360, 360/nproj)
        
    theta_tf = tf.convert_to_tensor(np.radians(theta), dtype=tf.float32)
    
    return(theta_tf)

def tf_tomo_radon(rec, ang, norm = False):
    
    '''
    Create the radon transform of an image
    Inputs:
        rec: 4D array corresponding to (1, npix, npix, 1)
        ang: 1D array corresponding to the projection angles
    '''
    
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tfa.image.rotate(img, -ang, interpolation = 'bilinear')
    sino = tf.reduce_sum(img, 1, name=None)
    if norm == True:
        sino = tf.image.per_image_standardization(sino)
    sino = tf.transpose(sino, [2, 0, 1])
    sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    return sino

def tf_tomo_bp(sino, ang, projmean = False, norm = False):
    
    '''
    Create the CT back projected image
    Inputs:
        sino: 4D array corresponding to (1, nproj, npix, 1)
        ang: 1D array corresponding to the projection angles
    '''
    d_tmp = sino.shape
    prj = tf.reshape(sino, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang, interpolation = 'bilinear')
    
    if projmean == True:
        bp = tf.reduce_mean(prj, 0)
    else:
        bp = tf.reduce_sum(prj, 0) * tf.convert_to_tensor(np.pi / (len(ang)), dtype='float32')
    
    if norm == True:
        bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], 1])
    return bp

def tf_ramp(detector_width):
    
    '''
    Creation of ramp filter
    Need to do this in tf
    '''
    
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width / 2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0] / 2.0)) * frequency_spacing)
                        
    return(tf_convert(filter_array.astype(np.float32)))

def tf_filt2D(d_sino):

    '''
    Reshape the 1D tomo reconstruction filter to 2D
    Inputs:
        d_sino: 1D array containing the two dimensions of the sinogram (nproj, npix)
    '''
    
    ft = np.array([tf_ramp(d_sino[1])])
    ft = tf.reshape(ft, [1, d_sino[1]])
    ft = tf.tile(ft, [d_sino[0], 1])
    
    return(ft)
    
def tf_tomo_fbp(sinogram, ang, ft=None):
    
    d_tmp = sinogram.shape
    soi = tf.reshape(sinogram, [d_tmp[1], d_tmp[2]])

    if ft is None:
        ft = tf_filt2D(d_tmp)

    sinogram_frequency = tf.signal.fft(tf.cast(soi, tf.complex64))
    filtered_sinogram_frequency = tf.multiply(sinogram_frequency, tf.cast(ft,dtype=tf.complex64))
    filtered_sinogram = tf.math.real(tf.signal.ifft(filtered_sinogram_frequency))   
    filtered_sinogram = tf.reshape(filtered_sinogram, [1, d_tmp[1], d_tmp[2], 1])

    rec = tf_tomo_bp(filtered_sinogram, ang)

    return(rec)

def tf_create_ramp_filter(s, ang):
    N1 = s.shape[1]
    freqs = np.linspace(-1, 1, N1)
    myFilter = np.abs( freqs )
    myFilter = np.tile(myFilter, (len(ang), 1))
    return(myFilter)

def tf_filter_sino(s, flt):

    sino_freq = tf.signal.fft(tf.cast(s, tf.complex64))
    sino_freq = tf.signal.fftshift(sino_freq, 1)
    filtered_freq = sino_freq * flt
    filtered_sino = tf.signal.ifftshift(filtered_freq,1)
    filtered_sino = tf.signal.ifft(filtered_sino)
    filtered_sinogram = tf.math.real(filtered_sino)
    return(filtered_sinogram)


def tf_mask_circle(img, npix=0):    
    # Mask everything outside the reconstruction circle
    sz = tf.math.floor(float(img.shape[1]))
    x = tf.range(0,sz)
    x = tf.repeat(x, int(sz))
    x = tf.reshape(x, (sz, sz))
    y = tf.transpose(x)
    xc = tf.math.round(sz/2)
    yc = tf.math.round(sz/2)
    r = tf.math.sqrt(((x-xc)**2 + (y-yc)**2))
    img = tf.where(r>sz/2 - npix,0.0,img[:,:,:,0])
    img = tf.reshape(img, (1, sz, sz, 1))
    return(img)

def tf_convert(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


def tf_Amatrix(A, gpu = True):

    '''
    Create the tf sparse A matrix and its transpose
    This can be improved
    '''

    Acoo = A.tocoo()

    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))

    Atf = tf.SparseTensor(indices, Acoo.data, Acoo.shape)

    ATtf = tf.sparse.transpose(Atf)

    return(Atf, ATtf)

def tf_Amatrix_sino(Atf, im, npr, ntr):

    '''
    Create sinogram using the A matrix
    '''

    stf = tf.sparse.sparse_dense_matmul(Atf, im)
    stf = tf.reshape(stf, (npr, ntr))

    return(stf)

def tf_Amatrix_rec(ATtf, s, ntr):

    '''
    Create reconstructed image using the A matrix
    '''

    rec = tf.sparse.sparse_dense_matmul(ATtf,s)
    rec = tf.reshape(rec, (ntr, ntr))

    return(rec)
