# -*- coding: utf-8 -*-
"""
Tensorflow functions for tomography
"""

import tensorflow as tf
import tensorflow_addons as tfa
import math as m
import numpy as np

def tomo_transf(im):
    return(tf.transpose(tf.reshape(im, (im.shape[0], im.shape[1], 1, 1)), (2, 0, 1, 3)))

def tomo_radon(rec, ang, norm = False):
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

def tomo_bp(sinoi, ang, projmean = False, norm = False):
    d_tmp = sinoi.shape
    prj = tf.reshape(sinoi, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang, interpolation = 'bilinear')
    
    if projmean == True:
        bp = tf.reduce_mean(prj, 0)
    else:
        bp = tf.reduce_sum(prj, 0)* tf.constant(m.pi) / (2 * len(ang))
    
    if norm == True:
        bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], 1])
    return bp

def ramp_tf(detector_width):
    
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
    return(convert(filter_array.astype(np.float32)))

def tomo_fbp(sinogram, ang):
    d_tmp = sinogram.shape
    soi = tf.reshape(sinogram, [d_tmp[1], d_tmp[2]])
    ft = np.array([ramp_tf(d_tmp[2])])
    ft = tf.reshape(ft, [1, d_tmp[2]])
    ft = tf.tile(ft, [d_tmp[1], 1])

    sinogram_frequency = tf.signal.fft(tf.cast(soi, tf.complex64))
    filtered_sinogram_frequency = tf.multiply(sinogram_frequency, tf.cast(ft,dtype=tf.complex64))
    filtered_sinogram = tf.math.real(tf.signal.ifft(filtered_sinogram_frequency))   
    filtered_sinogram = tf.reshape(filtered_sinogram, [1, d_tmp[1], d_tmp[2], 1])

    rec = tomo_bp(filtered_sinogram, ang)

    return(rec)

def create_ramp_filter(s, ang):
    N1 = s.shape[1]
    freqs = np.linspace(-1, 1, N1)
    myFilter = np.abs( freqs )
    myFilter = np.tile(myFilter, (len(ang), 1))
    return(myFilter)

def filter_sino(s, flt):

    sino_freq = tf.signal.fft(tf.cast(s, tf.complex64))
    sino_freq = tf.signal.fftshift(sino_freq, 1)
    filtered_freq = sino_freq * flt
    filtered_sino = tf.signal.ifftshift(filtered_freq,1)
    filtered_sino = tf.signal.ifft(filtered_sino)
    filtered_sinogram = tf.math.real(filtered_sino)
    return(filtered_sinogram)


def mask_circle(img, npix=0):    
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

def convert(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


def Amatrix_tf(A, gpu = True):

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

def Amatrix_sino(Atf, im, npr, ntr):

    '''
    Create sinogram using the A matrix
    '''

    stf = tf.sparse.sparse_dense_matmul(Atf, im)
    stf = tf.reshape(stf, (npr, ntr))

    return(stf)

def Amatrix_rec(ATtf, s, ntr):

    '''
    Create reconstructed image using the A matrix
    '''

    rec = tf.sparse.sparse_dense_matmul(ATtf,s)
    rec = tf.reshape(rec, (ntr, ntr))

    return(rec)