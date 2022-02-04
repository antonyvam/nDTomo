# -*- coding: utf-8 -*-
"""
Neural networks models

@author: Antony Vamvakeros and Hongyang Dong
"""

#%%

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import sin
import tensorflow_addons as tfa
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, UpSampling1D, Activation, Subtract, LeakyReLU, LayerNormalization, SpatialDropout2D, Average, Add, Input, concatenate, UpSampling2D, Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Cropping2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

    
def unet1D_5L(ntr, pad='same', skip=False):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, 1))
    
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(lowres_in)
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)

    conv2 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.1)(conv2)

    conv3 = Conv1D(64, 25, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.1)(conv3)

    conv4 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.1)(conv4)

    conv5 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.1)(conv5)

    up6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up6)

    merge6 = concatenate([conv4,up6], axis = 2)
    conv6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.1)(conv6)

    up7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up7)

    merge7 = concatenate([conv3,up7], axis = 2)
    conv7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.1)(conv7)

    up8 = Conv1D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up8)

    merge8 = concatenate([conv2,up8], axis = 2)
    conv8 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.1)(conv8)

    up9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up9)

    merge9 = concatenate([conv1,up9], axis = 2)
    conv9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(32, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.1)(conv9)

    conv10 = Conv1D(1, 1, activation = 'linear')(conv9)

    if skip == True:
        
        added = Add()([lowres_in, conv10])
        model = Model(lowres_in, added)

    else: 
        
        model = Model(lowres_in, conv10)

    return model

def autoencoder1D(ntr, pad='same'):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, 1))
    
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(lowres_in)
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)

    conv2 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.1)(conv2)

    conv3 = Conv1D(64, 25, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.1)(conv3)

    conv4 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.1)(conv4)

    conv5 = Conv1D(64, 25, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.1)(conv5)

    up6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up6)

    conv6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up6)
    conv6 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.1)(conv6)

    up7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up7)

    conv7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up7)
    conv7 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.1)(conv7)

    up8 = Conv1D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up8)

    conv8 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up8)
    conv8 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.1)(conv8)

    up9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling1D(size = 2)(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(up9)

    conv9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up9)
    conv9 = Conv1D(64, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(32, 25, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.1)(conv9)

    conv10 = Conv1D(1, 1, activation = 'linear')(conv9)

    model = Model(lowres_in, conv10)

    return model

def DnCNN(nt):
    
    inpt = Input(shape=(nt,nt,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    # x = Subtract()([inpt, x])   # input - noise
    
    # added = Add()([inpt, x])
    
    model = Model(inpt, x)
    
    return model

def recnet(np, nt):

    xi = Input(shape=( np, nt, 1))

    x = Flatten()(xi)

    x = Dense(500, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(500, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(500, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(nt * nt * 1, kernel_initializer='random_normal', activation='linear')(x)
    x = BatchNormalization()(x)

    x = Reshape((nt, nt, 1))(x)

    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)

    model = Model(xi, x)

    return(model)


def padcalc(npix, nlayers = 4):
        
    pads = [tf.convert_to_tensor(npix, dtype='float64')]
    
    for ii in range(1,nlayers):
    
        pads.append(tf.math.ceil(pads[ii-1]/2))
        
    pads = pads[::-1]
    
    return(pads)

def convblock(convl, nconvs = 3, filtnums= 64, kersz = 3, pad='same', dropout = 'Yes', batchnorm = 'No'):
        
    for ii in range(nconvs):
    
        convl = Conv2D(filters=filtnums, kernel_size=kersz, activation = 'relu', padding = pad,
                    kernel_initializer = 'he_normal')(convl)

    if batchnorm == 'Yes':
        convl = BatchNormalization()(convl)

    if dropout == 'Yes':
        convl = SpatialDropout2D(0.1)(convl)    
    
    return(convl)

def downblock(convl, nconvs = 3, filtnums= 64, kersz = 3, pad='same', dropout = 'Yes', batchnorm = 'No'):
    
    convl = Conv2D(filters=filtnums, kernel_size=kersz, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)
    convl = convblock(convl, nconvs = nconvs - 1, filtnums= filtnums, kersz = kersz, pad=pad, dropout = dropout, batchnorm = batchnorm)

    return(convl)    
    
def upblock(convl, padsize, filtnums= 64, pad='same'):
    
    convl = UpSampling2D(size = (2,2))(convl)

    convl = Conv2D(filters=filtnums, kernel_size=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)

    if padsize % 2 != 0:
        convl = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(convl)
    
    return(convl)

def DCNN2D(npix, nlayers = 4, net = 'unet', dlayer = 'No', skipcon = 'No', nconvs =3, filtnums= 64, kersz = 3, dropout = 'Yes', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear', pad='same'):

    '''
    
    Inputs:
        npix: number of pixels in the input image (assumes square image)
        nlayers: the depth of the CNN
        net: type of network, options are 'unet', 'autoencoder'
        dlayer: 'Yes/No' string; if a series of dense layers will be used after the most downscaled layer
        filtnums: number of filters to be used in the conv layers
        kersz: kernel size to be used in the conv layers
        dropout: 'Yes/No' string; if dropout (10%) will be used
        batchnorm: 'Yes/No' string; if batch normalisation will be used
        nconvs: number of convolutional layers per conv block
        actlayermid: the activation function used in all layers apart from the final layer
        actlayerfi: the activation function used in the final layer
        pad: padding type, default is 'same'
            
    '''
    pads = padcalc(npix, nlayers)
    
    image_in = Input(shape=(npix, npix,  1))
    convl = convblock(image_in, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    dconvs = [convl]
    
    for ii in range(nlayers):
        
        convl = downblock(convl, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
        dconvs.append(convl)
    
    if dlayer == 'Yes':
        
        convl = Flatten()(convl)
        
        for ii in range(3):
    
            convl = Dense(100, kernel_initializer='he_normal', activation = 'relu')(convl)
            convl = Dropout(0.1)(convl)
        
        convl = Dense(int(tf.math.ceil(pads[0] / 2)) * int(tf.math.ceil(pads[0] / 2)) * 8, kernel_initializer='he_normal', activation='relu')(convl)
        convl = Dropout(0.1)(convl)
        convl = Reshape((int(tf.math.ceil(pads[0] / 2)), int(tf.math.ceil(pads[0] / 2)), 8))(convl)    
        
    for ii in range(nlayers-1):

        up = upblock(convl, pads[ii], filtnums= filtnums, pad=pad)
        if net == 'unet':
            up = concatenate([dconvs[-(ii+2)],up], axis = 3)
        convl = convblock(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
    
    up = upblock(convl, pads[-1], filtnums= filtnums, pad=pad)
    if net == 'unet':
        up = concatenate([dconvs[0],up], axis = 3)
    convl = convblock(convl=up, nconvs = 2, dropout = 'No', batchnorm= 'No')

    convl = convblock(convl, nconvs = 1, filtnums= 32, kersz = 2, pad=pad, dropout=dropout, batchnorm=batchnorm)

    convl = Conv2D(1, 1, activation = actlayerfi)(convl)

    if skipcon == 'Yes':
        
        added = Add()([image_in, convl])
        model = Model(image_in, added)

    else:
    
        model = Model(image_in, convl)

    return model