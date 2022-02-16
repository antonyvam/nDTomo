# -*- coding: utf-8 -*-
"""
Neural networks models

@author: Antony Vamvakeros and Hongyang Dong
"""

#%%

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, UpSampling1D, Activation, Subtract, LeakyReLU, LayerNormalization, SpatialDropout2D, Average, Add, Input, concatenate, UpSampling2D, Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Cropping2D


def DnCNN(npix, nlayers = 15, skip='Yes', filts = 64):
    
    '''
    A simple 2D deep convolutional neural network having muptiple conv2D layers in series
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
        nlayers: number of conv2D layers
        skip: residual learning or not i.e. if the network uses a skip connection
        filts: number of filters in the conv2D layers
    '''
    
    im = Input(shape=(npix[0],npix[1],1))

    x = Conv2D(filters=filts, kernel_size=3, padding='same', activation='relu')(im)

    for i in range(nlayers):
        x = Conv2D(filters=filts, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    x = Conv2D(filters=1, kernel_size=3, padding='same')(x)
    
    if skip == 'Yes':
        added = Add()([im, x])
        model = Model(im, added)
    else:
        model = Model(im, x)
    
    return model

def recnet(npix):

    '''
    Image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
    
    '''
    
    nx, ny = npix
    
    xi = Input(shape=( nx, ny, 1))

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
    x = Dense(ny * ny * 1, kernel_initializer='random_normal', activation='linear')(x)
    x = BatchNormalization()(x)

    x = Reshape((ny, ny, 1))(x)

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


def convblock1D(convl, nconvs = 3, filtnums= 64, kersz = 25, pad='same', dropout = 'Yes', batchnorm = 'No'):
        
    for ii in range(nconvs):
    
        convl = Conv1D(filters=filtnums, kernel_size=kersz, activation = 'relu', padding = pad,
                    kernel_initializer = 'he_normal')(convl)

    if batchnorm == 'Yes':
        convl = BatchNormalization()(convl)

    if dropout == 'Yes':
        convl = Dropout(0.1)(convl)    
    
    return(convl)

def downblock1D(convl, nconvs = 3, filtnums= 64, kersz = 25, pad='same', dropout = 'Yes', batchnorm = 'No'):
    
    convl = Conv1D(filters=filtnums, kernel_size=kersz, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)
    convl = convblock1D(convl, nconvs = nconvs - 1, filtnums= filtnums, kersz = kersz, pad=pad, dropout = dropout, batchnorm = batchnorm)

    return(convl)    
    
def upblock1D(convl, padsize, filtnums= 64, pad='same'):
    
    convl = UpSampling1D(size = (2))(convl)

    convl = Conv1D(filters=filtnums, kernel_size=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)

    if padsize % 2 != 0:
        convl = tf.keras.layers.Cropping1D(
                cropping=(1, 0))(convl)
    
    return(convl)

def DCNN1D(npix, nlayers = 4, net = 'unet', dlayer = 'No', skipcon = 'No', nconvs =3, filtnums= 64, kersz = 25, dropout = 'Yes', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear', pad='same'):

    '''
    1D Deep Convolutional Neural Network
    
    Inputs:
        npix: number of pixels in the input 2D spectrum
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
    
    spectrum_in = Input(shape=(npix, 1))
    
    convl = convblock1D(spectrum_in, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    dconvs = [convl]
    
    for ii in range(nlayers):
        
        convl = downblock1D(convl, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
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

        print(ii)
        up = upblock1D(convl, pads[ii], filtnums= filtnums, pad=pad)
        if net == 'unet':
            up = concatenate([dconvs[-(ii+2)],up], axis = 2)
        convl = convblock1D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
    
    up = upblock1D(convl, pads[-1], filtnums= filtnums, pad=pad)
    if net == 'unet':
        up = concatenate([dconvs[0],up], axis = 2)
    convl = convblock1D(convl=up, nconvs = 2, dropout = 'No', batchnorm= 'No')

    convl = convblock1D(convl, nconvs = 1, filtnums= 32, kersz = 25, pad=pad, dropout=dropout, batchnorm=batchnorm)

    convl = Conv1D(1, 1, activation = actlayerfi)(convl)

    if skipcon == 'Yes':
        
        added = Add()([spectrum_in, convl])
        model = Model(spectrum_in, added)

    else:
    
        model = Model(spectrum_in, convl)

    return model

def Dense1D(npix, nlayers = 4, nodes = [100, 75, 50, 25], dropout = 'No', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear'):

    '''
    1D Deep Convolutional Neural Network
    
    Inputs:
        npix: number of pixels in the input 2D spectrum
        dropout: 'Yes/No' string; if dropout (10%) will be used
        batchnorm: 'Yes/No' string; if batch normalisation will be used
        actlayermid: the activation function used in all layers apart from the final layer
        actlayerfi: the activation function used in the final layer
            
    '''
    
    spectrum_in = Input(shape=(npix, 1))
    
    dlayer = Flatten()(spectrum_in)
    
    # Downscaling
    for ii in range(nlayers):

        dlayer = Dense(nodes[ii], kernel_initializer='he_normal', activation = actlayermid)(dlayer)
        
        if batchnorm == 'Yes':
            dlayer = BatchNormalization()(dlayer)
    
        if dropout == 'Yes':
            dlayer = Dropout(0.1)(dlayer)   
    
    # Upscaling
    for ii in range(1,nlayers-1):

        dlayer = Dense(nodes[-ii-1], kernel_initializer='he_normal', activation = actlayermid)(dlayer)
        
        if batchnorm == 'Yes':
            dlayer = BatchNormalization()(dlayer)
    
        if dropout == 'Yes':
            dlayer = Dropout(0.1)(dlayer)       
    
    dlayer = Dense(nodes[0], kernel_initializer='he_normal', activation = actlayerfi)(dlayer)
    
    if batchnorm == 'Yes':
        dlayer = BatchNormalization()(dlayer)

    if dropout == 'Yes':
        dlayer = Dropout(0.1)(dlayer)           

    model = Model(spectrum_in, dlayer)

    return model

def convblock2D(convl, nconvs = 3, filtnums= 64, kersz = 3, pad='same', dropout = 'Yes', batchnorm = 'No'):
        
    for ii in range(nconvs):
    
        convl = Conv2D(filters=filtnums, kernel_size=kersz, activation = 'relu', padding = pad,
                    kernel_initializer = 'he_normal')(convl)

    if batchnorm == 'Yes':
        convl = BatchNormalization()(convl)

    if dropout == 'Yes':
        convl = SpatialDropout2D(0.1)(convl)    
    
    return(convl)

def downblock2D(convl, nconvs = 3, filtnums= 64, kersz = 3, pad='same', dropout = 'Yes', batchnorm = 'No'):
    
    convl = Conv2D(filters=filtnums, kernel_size=kersz, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)
    convl = convblock2D(convl, nconvs = nconvs - 1, filtnums= filtnums, kersz = kersz, pad=pad, dropout = dropout, batchnorm = batchnorm)

    return(convl)    
    
def upblock2D(convl, padsize, filtnums= 64, pad='same'):
    
    convl = UpSampling2D(size = (2,2))(convl)

    convl = Conv2D(filters=filtnums, kernel_size=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)

    if padsize % 2 != 0:
        convl = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(convl)
    
    return(convl)

def DCNN2D(npix, nlayers = 4, net = 'unet', dlayer = 'No', skipcon = 'No', nconvs =3, filtnums= 64, kersz = 3, dropout = 'Yes', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear', pad='same', dense_layers = 'Default', dlayers = None):

    '''
    2D Deep Convolutional Neural Network
    
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
        dense_layers: 'Default/Custom' string; if 'Custom', then the use has to pass a list containing the number of nodes per dense layer
    '''
    pads = padcalc(npix, nlayers)
    
    image_in = Input(shape=(npix, npix,  1))
    convl = convblock2D(image_in, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    dconvs = [convl]
    
    for ii in range(nlayers):
        
        convl = downblock2D(convl, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
        dconvs.append(convl)
    
    if dlayer == 'Yes' and dense_layers == 'Default':
        
        convl = Flatten()(convl)
        
        for ii in range(3):
    
            convl = Dense(100, kernel_initializer='he_normal', activation = actlayermid)(convl)

            if batchnorm == 'Yes':
                convl = BatchNormalization()(convl)
        
            if dropout == 'Yes':
                convl = Dropout(0.1)(convl)                 
        
        convl = Dense(int(tf.math.ceil(pads[0] / 2)) * int(tf.math.ceil(pads[0] / 2)) * 8, kernel_initializer='he_normal', activation=actlayermid)(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
    
        if dropout == 'Yes':
            convl = Dropout(0.1)(convl)     
        convl = Reshape((int(tf.math.ceil(pads[0] / 2)), int(tf.math.ceil(pads[0] / 2)), 8))(convl)    
    
    elif dlayer == 'Yes' and dense_layers == 'Custom':
        
        convl = Flatten()(convl)
        
        for ii in range(len(dlayers)):
    
            convl = Dense(dlayers[ii], kernel_initializer='he_normal', activation = actlayermid)(convl)

            if batchnorm == 'Yes':
                convl = BatchNormalization()(convl)
        
            if dropout == 'Yes':
                convl = Dropout(0.1)(convl)                 
        
        convl = Dense(int(tf.math.ceil(pads[0] / 2)) * int(tf.math.ceil(pads[0] / 2)) * 8, kernel_initializer='he_normal', activation=actlayermid)(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
    
        if dropout == 'Yes':
            convl = Dropout(0.1)(convl)     
        convl = Reshape((int(tf.math.ceil(pads[0] / 2)), int(tf.math.ceil(pads[0] / 2)), 8))(convl)           
    
    for ii in range(nlayers-1):

        up = upblock2D(convl, pads[ii], filtnums= filtnums, pad=pad)
        if net == 'unet':
            up = concatenate([dconvs[-(ii+2)],up], axis = 3)
        convl = convblock2D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
    
    up = upblock2D(convl, pads[-1], filtnums= filtnums, pad=pad)
    if net == 'unet':
        up = concatenate([dconvs[0],up], axis = 3)
    convl = convblock2D(convl=up, nconvs = 2, dropout = 'No', batchnorm= 'No')

    convl = convblock2D(convl, nconvs = 1, filtnums= 32, kersz = 2, pad=pad, dropout=dropout, batchnorm=batchnorm)

    convl = Conv2D(1, 1, activation = actlayerfi)(convl)

    if skipcon == 'Yes':
        
        added = Add()([image_in, convl])
        model = Model(image_in, added)

    else:
    
        model = Model(image_in, convl)

    return model


def Dense2D(npix, nlayers = 4, nodes = [100, 75, 50, 25], dropout = 'No', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear',):

    '''
    2D Deep Convolutional Neural Network
    
    Inputs:
        npix: number of pixels in the input 2D spectrum
        dropout: 'Yes/No' string; if dropout (10%) will be used
        batchnorm: 'Yes/No' string; if batch normalisation will be used
        actlayermid: the activation function used in all layers apart from the final layer
        actlayerfi: the activation function used in the final layer
            
    '''
    
    image_in = Input(shape=(npix, npix,  1))
    
    dlayer = Flatten()(image_in)
    
    # Downscaling
    for ii in range(nlayers):

        dlayer = Dense(nodes[ii], kernel_initializer='he_normal', activation = actlayermid)(dlayer)
        
        if batchnorm == 'Yes':
            dlayer = BatchNormalization()(dlayer)
    
        if dropout == 'Yes':
            dlayer = Dropout(0.1)(dlayer)   
    
    # Upscaling
    for ii in range(1,nlayers):

        dlayer = Dense(nodes[-ii-1], kernel_initializer='he_normal', activation = actlayermid)(dlayer)
        
        if batchnorm == 'Yes':
            dlayer = BatchNormalization()(dlayer)
    
        if dropout == 'Yes':
            dlayer = Dropout(0.1)(dlayer)       
    
    dlayer = Dense(npix*npix, kernel_initializer='he_normal', activation = actlayerfi)(dlayer)
    
    if batchnorm == 'Yes':
        dlayer = BatchNormalization()(dlayer)

    if dropout == 'Yes':
        dlayer = Dropout(0.1)(dlayer)           
        
    dlayer = Reshape((npix, npix, 1))(dlayer)  

    model = Model(image_in, dlayer)

    return model