# -*- coding: utf-8 -*-
"""
Neural networks models

@authors: Antony Vamvakeros and Hongyang Dong
"""

#%%

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Conv1D, UpSampling1D, Activation, Subtract, LeakyReLU, LayerNormalization, SpatialDropout2D, Average, Add, Input, concatenate, UpSampling2D, Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Cropping2D
from numpy import mod, ceil

def SD2I(npix, factor=8, upsample = True):

    '''
    SD2I image reconstruction network with upsampling
    
    Inputs:
        npix: number of pixels in the image per each dimension
    
    '''
    
    xi = Input(shape=(1,))
    x = Flatten()(xi)
    
    if upsample:
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(int(ceil(npix / 4)) * int(ceil(npix / 4)) * factor, kernel_initializer='random_normal', activation='linear')(x)
        
        x = Reshape((int(ceil(npix / 4)), int(ceil(npix / 4)), factor))(x)   
        
        x = UpSampling2D(size = (2,2))(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        
        x = UpSampling2D(size = (2,2))(x)

        x = Cropping2D(cropping=((1, 0), (1, 0)))(x)

        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

        x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    else:
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(npix * npix * factor, kernel_initializer='random_normal', activation='linear')(x)
        
        x = Reshape((npix, npix, factor))(x)
        
        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

        x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
        x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)

    model = Model(xi, x)

    return(model)

def Automap(npix, npr):

    '''
    Automap image reconstruction network
    
    Inputs:
        npix: number of pixels in the reconstructed image per each dimension (number of detector elements in sinograms)
        npr: number of tomographic angles (projections)
    
    '''
    

    xi = Input(shape=(npr,npix))
    x = Flatten()(xi)
    
    x = Dense(2*npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)
    
    x = Reshape((int(npix // 1), int(npix // 1), 1))(x)   
    
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    
    model = Model(xi, x)

    return(model)

def DnCNN(nlayers = 15, skip='Yes', filts = 64):
    
    '''
    A simple 2D deep convolutional neural network having muptiple conv2D layers in series
    Inputs:
        nlayers: number of conv2D layers
        skip: residual learning or not i.e. if the network uses a skip connection
        filts: number of filters in the conv2D layers
    '''
    
    im = Input(shape=(None,None,1))

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

    x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, kernel_initializer='random_normal', activation='relu')(x)
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

def recnet_single(npix):

    '''
    Image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
    
    '''
    
    nx, ny = npix
    
    xi = Input(shape=(1,1))
    x = Flatten()(xi)
    
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(nx * ny * 1, kernel_initializer='random_normal', activation='linear')(x)
    x = BatchNormalization()(x)

    x = Reshape((nx, ny, 1))(x)

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

def recnet_single_conv(npix, factor = 1, dropout = 'No', batchnorm = 'No', actlayerfi='linear', pad='same'):

    '''
    Image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
    
    '''
        
    nx, ny = npix
    
    m = mod(nx,2)
    
    xi = Input(shape=(1,))
    x = Flatten()(xi)
    
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)
    x = Dense(64, kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)
    x = Dense(int(nx / 4) * int(ny / 4) * factor, kernel_initializer='he_normal', activation='linear')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    
    x = Reshape((int(nx / 4), int(ny / 4), factor))(x)   
    
    x = UpSampling2D(size = (2,2))(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = SpatialDropout2D(0.1)(x)    
    
    x = UpSampling2D(size = (2,2))(x)
    
    if m>0:
        x = Cropping2D(cropping=((1, 0), (1, 0)))(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = SpatialDropout2D(0.1)(x)         

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



def DCNN2D(nx, ny, nlayers = 4, net = 'unet', dlayer = 'No', skipcon = 'No', nconvs =3, filtnums= 64, kersz = 3, dropout = 'Yes', batchnorm = 'No', 
                  actlayermid = 'relu', actlayerfi = 'linear', pad='same', dense_layers = 'Default', dlayers = None):

    '''
    2D Deep Convolutional Neural Network
    
    Inputs:
        nx, ny: number of pixels in the input image (rows, columns)
        nomega: number of pixels in the second dimension (e.g. number of projections for sinograms in tomography)
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
        dense_layers: 'Default/Custom' string; if 'Custom', then the use has to pass a list conpixaining the number of nodes per dense layer
    '''
    
    pads_x = padcalc(nx, nlayers)
    pads_y = padcalc(ny, nlayers)
    
    image_in = Input(shape=(nx, ny,  1))
    convl = convblock2D(image_in, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    dconvs = [convl]
    
    for ii in range(nlayers):
        
        convl = downblock2D(convl, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
        dconvs.append(convl)
    
    if dlayer == 'Yes' and dense_layers == 'Default':
        
        convl = Conv2D(1, 1, activation = actlayermid)(convl)
        
        convl = Flatten()(convl)
        
        for ii in range(3):
    
            convl = Dense(100, kernel_initializer='he_normal', activation = actlayermid)(convl)

            if batchnorm == 'Yes':
                convl = BatchNormalization()(convl)
        
            if dropout == 'Yes':
                convl = Dropout(0.1)(convl)                 
        
        convl = Dense(int(nx / 4) * int(nx / 4) * 1, kernel_initializer='he_normal', activation=actlayermid)(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
    
        if dropout == 'Yes':
            convl = Dropout(0.1)(convl)     
        convl = Reshape((int(nx / 4) , int(nx / 4), 1))(convl)    

        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
        if dropout == 'Yes':
            convl = SpatialDropout2D(0.1)(convl)    
        
        for ii in range(nlayers-2):
    
            up = upblock2D(convl, [pads_x[ii], pads_y[ii]] , filtnums= filtnums, pad=pad)
            if net == 'unet':
                up = concatenate([dconvs[-(ii+2)],up], axis = 3)
            convl = convblock2D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    elif dlayer == 'Yes' and dense_layers == 'Custom':
        
        convl = Conv2D(1, 1, activation = actlayermid)(convl)
        
        convl = Flatten()(convl)
        
        for ii in range(len(dlayers)):
    
            convl = Dense(dlayers[ii], kernel_initializer='he_normal', activation = actlayermid)(convl)
    
            if batchnorm == 'Yes':
                convl = BatchNormalization()(convl)
        
            if dropout == 'Yes':
                convl = Dropout(0.1)(convl)                 
        
        convl = Dense(int(nx / 4) * int(nx / 4) * 1, kernel_initializer='he_normal', activation=actlayermid)(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
    
        if dropout == 'Yes':
            convl = Dropout(0.1)(convl)     
        convl = Reshape((int(nx / 4) , int(nx / 4), 1))(convl)    

        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        convl = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(convl)
        if batchnorm == 'Yes':
            convl = BatchNormalization()(convl)
        if dropout == 'Yes':
            convl = SpatialDropout2D(0.1)(convl)    
        
        for ii in range(nlayers-2):
    
            up = upblock2D(convl, [pads_x[ii], pads_y[ii]], filtnums= filtnums, pad=pad)
            if net == 'unet':
                up = concatenate([dconvs[-(ii+2)],up], axis = 3)
            convl = convblock2D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
         
    else:
        
        for ii in range(nlayers-1):
    
            up = upblock2D(convl, [pads_x[ii], pads_y[ii]], filtnums= filtnums, pad=pad)
            if net == 'unet':
                up = concatenate([dconvs[-(ii+2)],up], axis = 3)
            convl = convblock2D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)        
    
    up = upblock2D(convl, [pads_x[-1], pads_y[-1]], filtnums= filtnums, pad=pad)
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
    
def upblock2D(convl, padsizes, filtnums= 64, pad='same'):
    
    convl = UpSampling2D(size = (2,2))(convl)

    convl = Conv2D(filters=filtnums, kernel_size=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(convl)

    if padsizes[0] % 2 != 0:
        convl = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (0, 0)), data_format=None)(convl)

    if padsizes[1] % 2 != 0:
        convl = tf.keras.layers.Cropping2D(
                cropping=((0, 0), (1, 0)), data_format=None)(convl)
    
    return(convl)
    
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
	

def edsr_block(x, scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):

    '''
    An EDSR model block that can be used as part of a larger architecture
    Original EDSR model taken from https://github.com/krasserm/super-resolution/
    Inputs:
        x: the previous layer
    '''

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    return(x)


def res_block(x_in, filters, scaling, batchnorm = 'No', dropout = 'No'):

    '''
    Residual block taken from https://github.com/krasserm/super-resolution/
    '''
	
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)

    if dropout == 'Yes':
        x = Dropout(0.1)(x)         
    return x


def upsample(x, scale, num_filters):

    '''
    Upsample layer taken from https://github.com/krasserm/super-resolution/
    '''
	
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3)
    elif scale == 4:
        x = upsample_1(x, 2)
        x = upsample_1(x, 2)

    return x
	
def pixel_shuffle(scale):

    '''
    Pixel shuffle function taken from https://github.com/krasserm/super-resolution/
    '''
	
    return lambda x: tf.nn.depth_to_space(x, scale)

def DCNN2D_block(npix, image_in, pads, nlayers = 4, net = 'unet', dlayer = 'No', skipcon = 'No', nconvs =3, filtnums= 64, kersz = 3, dropout = 'Yes', batchnorm = 'No', 
                  actlayermid = 'relu', pad='same', dense_layers = 'Default', dlayers = None):
    
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

    return convl

def DCNN2D_dual(npix, nlayers=4, net = 'unet', dropout = 'No', batchnorm = 'No', 
                nconvs =3, filtnums= 64,
                actlayerfi = 'linear', skipcon = 'No', actlayermid = 'relu'):
    
    pads = padcalc(npix, nlayers)
    
    image_in1 = Input(shape=(npix, npix,  1))    
    convl1 = DCNN2D_block(npix, image_in1, pads, nlayers = nlayers, net = net, skipcon = skipcon, nconvs =nconvs, filtnums= filtnums, 
                         kersz = 3, dropout = dropout, batchnorm = batchnorm, actlayermid = actlayermid)    
    
    image_in2 = Input(shape=(npix, npix,  1))    
    convl2 = DCNN2D_block(npix, image_in2, pads, nlayers = nlayers, net = net, skipcon = skipcon, nconvs =nconvs, filtnums= filtnums, 
                         kersz = 3, dropout = dropout, batchnorm = batchnorm, actlayermid = actlayermid) 
    
    convl = concatenate([convl1,convl2], axis = 3)
    
    convl = convblock2D(convl, nconvs = 1, filtnums= 32, kersz = 2, pad='same', dropout=dropout, batchnorm=batchnorm)

    convl = Conv2D(1, 1, activation = actlayerfi)(convl)

    if skipcon == 'Yes':
        
        added = Add()([(image_in1+image_in2)/2, convl])
        model = Model(inputs = [image_in1, image_in2],outputs = added)

    else:
    
        model = Model(inputs = [image_in1, image_in2],outputs = convl)
    
        
    return model    




def Automap(npix, pad='same', actlayerfi='linear', batchnorm = 'No', dropout = 'No'):

    '''
    Image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
    
    '''
    
    xi = Input(shape=(npix[0],npix[1],1))
    x = Flatten()(xi)
    
    x = Dense(2*npix[0]*npix[1], kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)       
    x = Dense(npix[0]*npix[1], kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)  
    x = Dense(npix[0]*npix[1], kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x)  
    
    x = Reshape((int(npix[0] // 1), int(npix[1] // 1), 1))(x)   
    
    x = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)

    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    model = Model(xi, x)

    return(model)

    
def GANrec_dropout(npix, pad='same', actlayerfi='linear', batchnorm = 'No', dropout = 'No'):

    '''
    Image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension; it is a list [npixs_x, npixs_y]
    
    '''
    
    
    xi = Input(shape=(npix[0],npix[1],1))
    x = Flatten()(xi)
    
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x) 
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x) 
    x = Dense(npix[0]*npix[1], kernel_initializer='random_normal', activation='relu')(x)
    if batchnorm == 'Yes':
        x = BatchNormalization()(x)
    if dropout == 'Yes':
        x = Dropout(0.1)(x) 

    
    x = Reshape((int(npix[0] // 1), int(npix[1] // 1), 1))(x)   
    
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    model = Model(xi, x)

    return(model)


def GANrec(npix):

    '''
    GANrec image reconstruction network
    
    Inputs:
        npix: number of pixels in the image per each dimension
    
    '''
    
    xi = Input(shape=(npix,npix,1))
    x = Flatten()(xi)
    
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(256, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(npix*npix, kernel_initializer='random_normal', activation='relu')(x)

    
    x = Reshape((int(npix // 1), int(npix // 1), 1))(x)   
    
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')(x)
    x = Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'linear')(x)
    
    model = Model(xi, x)

    return(model)


class Discriminator(tf.keras.Model):
    
    def __init__(self, npix, npr):
        super(Discriminator, self).__init__()

        self.conv_1 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_2 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_3 = Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.conv_4 = Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding = 'same', kernel_initializer='random_normal', activation = 'relu')
        self.reshape = Reshape((npix * npr * 512, ))

    def call(self, inputs):
        tf.keras.backend.set_floatx('float32')
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return self.reshape(x)


def DFCNN2D(nx, ny, nlayers = 4, net = 'unet', skipcon = 'No', 
            nconvs =3, filtnums= 64, kersz = 3, dropout = 'Yes', batchnorm = 'No', 
            actlayermid = 'relu', actlayerfi = 'linear', pad='same'):

    '''
    2D Deep Fully Convolutional Neural Network
    
    Inputs:
        npix: number of pixels in the input image
        nomega: number of pixels in the second dimension (e.g. number of projections for sinograms in tomography)
        nlayers: the depth of the CNN
        net: type of network, options are 'unet', 'autoencoder'
        filtnums: number of filters to be used in the conv layers
        kersz: kernel size to be used in the conv layers
        dropout: 'Yes/No' string; if dropout (10%) will be used
        batchnorm: 'Yes/No' string; if batch normalisation will be used
        nconvs: number of convolutional layers per conv block
        actlayermid: the activation function used in all layers apart from the final layer
        actlayerfi: the activation function used in the final layer
        pad: padding type, default is 'same'
    '''
    pads_x = padcalc(nx, nlayers)
    pads_y = padcalc(ny, nlayers)
    
    
    image_in = Input(shape=(None, None,  1))
    convl = convblock2D(image_in, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)

    dconvs = [convl]
    
    for ii in range(nlayers):
        
        convl = downblock2D(convl, nconvs = 3, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)
        dconvs.append(convl)
    

    for ii in range(nlayers-1):

        up = upblock2D(convl, [pads_x[ii], pads_y[ii]], filtnums= filtnums, pad=pad)
        if net == 'unet':
            up = concatenate([dconvs[-(ii+2)],up], axis = 3)
        convl = convblock2D(convl=up, nconvs = 2, filtnums=filtnums, kersz=kersz, pad=pad, dropout=dropout, batchnorm=batchnorm)        
    
    up = upblock2D(convl, [pads_x[-1], pads_y[-1]], filtnums= filtnums, pad=pad)
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

def CNN1D(nlayers = 4, skip=False, filts = 64, kernel_size=10):
    
    '''
    A simple 1D deep convolutional neural network having muptiple conv1D layers in series
    Inputs:
        nlayers: number of conv1D layers
        skip: residual learning or not i.e. if the network uses a skip connection
        filts: number of filters in the conv1D layers
    '''
    
    data = Input(shape=(None,1))

    x = Conv1D(filters=filts, kernel_size=kernel_size, padding='same', activation='relu')(data)

    for i in range(nlayers):
        x = Conv1D(filters=filts, kernel_size=kernel_size, padding='same')(x)
        x = Conv1D(filters=filts, kernel_size=kernel_size, padding='same', activation='relu')(x)

    x = Conv1D(filters=1, kernel_size=kernel_size, padding='same', activation='linear')(x)
    
    if skip:
        added = Add()([data, x])
        model = Model(data, added)
    else:
        model = Model(data, x)
    
    return model
