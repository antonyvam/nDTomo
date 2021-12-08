# -*- coding: utf-8 -*-
"""
Tensorflow functions for tomography
"""

import tensorflow as tf
import tensorflow_addons as tfa

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

def tomo_bp(sinoi, ang, norm = False):
    d_tmp = sinoi.shape
    prj = tf.reshape(sinoi, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang, interpolation = 'bilinear')
    bp = tf.reduce_mean(prj, 0)
    if norm == True:
        bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], bp.shape[2]])
    return bp

def convert(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def mask_circle(img, npix):    
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
    img = tf.resha