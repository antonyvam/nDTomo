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
    intersection = tf.keras.sum(tf.keras.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (tf.keras.sum(tf.keras.square(y_true),-1) + tf.keras.sum(tf.keras.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def penalty(img):

    image = img[:, :, 0]
    mean = tf.zeros(image.shape, dtype=tf.dtypes.float32, name=None)
    ind0 = image.shape[0]
    ind1 = image.shape[1]
    image_pad1 = tf.pad(image, tf.constant([[1, 1,], [1, 1]]), "REFLECT")
    image_pad2 = tf.pad(image, tf.constant([[2, 2,], [2, 2]]), "REFLECT")
    mean_list = []
    for i in range(3):
        for j in range(3):
            # if i != 1 and j != 1:
            mean += image_pad1[i:i+ind0, j:j+ind1]
            mean_list.append(image_pad1[i:i+ind0, j:j+ind1])
    mean_list = tf.stack(mean_list)

    mean = mean/9
    mean_pad = tf.pad(mean, tf.constant([[1, 1], [1, 1]]), "REFLECT")
    contribution = tf.zeros(image.shape, dtype=tf.dtypes.float32, name=None)

    individual = []
    for i in range(3):
        for j in range(3):
            # if i != 1 and j != 1:
            individual.append(mean_pad[i:i+ind0, j:j+ind1])
    individual = tf.stack(individual)
    individual_min = tf.math.reduce_min(individual, axis=0, keepdims=False, name=None)

    individual = individual-individual_min
    contribution = tf.reduce_sum(individual, axis = 0)
    contribution = individual / contribution

    #normalize and take the inverse
    contribution = contribution - contribution[4]
    contribution = tf.abs(contribution)
    contribution = tf.clip_by_value(contribution, 0.001, 1)
    index_list = [0,1,2,3,5,6,7,8]
    contribution = 1/ (tf.gather(contribution, index_list))

    #normalize again
    contribution_min = tf.math.reduce_min(contribution, axis=0, keepdims=False, name=None)
    contribution = contribution - contribution_min
    contribution_sum = tf.reduce_sum(contribution, axis = 0)
    contribution = contribution / contribution_sum
 
    mean_list = tf.gather(mean_list, index_list)
    mean_contrib = tf.reduce_sum(mean_list * contribution, axis = 0)
    # mean_contrib += mean
    # mean_contrib /= 2
    # for i in range(5):
    #     for j in range(5):
    #         mean += image_pad2[i:i+ind0, j:j+ind1]/5
    # mean = mean/15
    # mean =  0
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         mean += tf.math.reduce_mean(image[i-1:i+2, j-1:j+2]) - image[i,j]

    # return mean_contrib
    # return mean_contrib
    return tf.reduce_mean(tf.abs(mean_contrib - image)**2)