# -*- coding: utf-8 -*-
"""

Loss functions

"""

import tensorflow as tf

def ssim_loss(y_true, y_pred):
    return(1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0)))

def psnr_loss(y_true, y_pred):
    """
    The cost function by computing the psnr.
    """
    return 1/(10.0 * tf.math.log(1.0 / (tf.reduce_mean(tf.math.square(y_pred - y_true)))) / tf.math.log(10.0))

def root_mean_squared_error_loss(y_true, y_pred):
    return tf.math.sqrt(tf.reduce_mean(tf.math.square(y_pred - y_true)))

def ssim_mae_loss(y_true, y_pred):
    return((1-0.84)*tf.reduce_mean(tf.keras.losses.MAE(y_pred, y_true)) + 0.84*(1 - tf.reduce_mean(tf.image.ssim(y_pred, y_true, 2.0))))

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = tf.keras.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.keras.sum(tf.keras.square(y_true),-1) + tf.keras.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)