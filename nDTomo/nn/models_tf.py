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

def unet2D_small(ntr, skip = False, pad='same'):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, ntr, 1))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(lowres_in)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)

    conv2 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = SpatialDropout2D(0.1)(conv2)

    conv3 = Conv2D(64, 3, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = SpatialDropout2D(0.1)(conv3)

    conv4 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = SpatialDropout2D(0.1)(conv4)

    conv5 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = SpatialDropout2D(0.1)(conv5)

    up6 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up6)

    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SpatialDropout2D(0.1)(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up7)

    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SpatialDropout2D(0.1)(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SpatialDropout2D(0.1)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up9)

    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SpatialDropout2D(0.1)(conv9)

    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    if skip:
        
        added = Add()([lowres_in, conv10])
        model = Model(lowres_in, added)        
        
    else:
        
        model = Model(lowres_in, conv10)

    return model
    

def autoencoder2D_5L(ntr, pad='same'):

    up_9_pad = ntr
    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)

    image_in = Input(shape=(ntr, ntr,  1))

    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(image_in)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)

    conv2 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = SpatialDropout2D(0.1)(conv2)

    conv3 = Conv2D(64, 3, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = SpatialDropout2D(0.1)(conv3)

    conv4 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = SpatialDropout2D(0.1)(conv4)

    conv5 = Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = SpatialDropout2D(0.1)(conv5)

    up6 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up6)

    conv6 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = SpatialDropout2D(0.1)(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up7)

    conv7 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = SpatialDropout2D(0.1)(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up8)

    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = SpatialDropout2D(0.1)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(up9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = SpatialDropout2D(0.1)(conv9)

    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    model = Model(image_in, conv10)

    return model


def autoencoder2D_DL(npr, ntr, pad='same'):

    sinogram_in = Input(shape=(npr, ntr,  1))

    up_9_pad_sino = ntr
    up_8_pad_sino = np.ceil(up_9_pad_sino / 2)
    up_7_pad_sino = np.ceil(up_8_pad_sino / 2)
    up_6_pad_sino = np.ceil(up_7_pad_sino / 2)
    
    conv1_1_sino = Conv2D(64, 3, padding = pad,
            kernel_initializer = 'he_normal')(sinogram_in)
    conv1_1l_sino = LeakyReLU(alpha=0.2)(conv1_1_sino)
    conv1_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv1_1l_sino)
    conv1_2l_sino = LeakyReLU(alpha=0.2)(conv1_2_sino)            
    conv1_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv1_2l_sino)
    conv1_3l_sino = LeakyReLU(alpha=0.2)(conv1_3_sino)  
    conv1_dr_sino = SpatialDropout2D(0.1)(conv1_3l_sino)

    conv2_1_sino = Conv2D(64, 3, strides=2, padding = pad,
            kernel_initializer = 'he_normal')(conv1_dr_sino)
    conv2_1l_sino = LeakyReLU(alpha=0.2)(conv2_1_sino)
    conv2_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv2_1l_sino)
    conv2_2l_sino = LeakyReLU(alpha=0.2)(conv2_2_sino)            
    conv2_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv2_2l_sino)
    conv2_3l_sino = LeakyReLU(alpha=0.2)(conv2_3_sino)  
    conv2_dr_sino = SpatialDropout2D(0.1)(conv2_3l_sino)

    conv3_1_sino = Conv2D(64, 3, strides=2, padding = pad,
            kernel_initializer = 'he_normal')(conv2_dr_sino)
    conv3_1l_sino = LeakyReLU(alpha=0.2)(conv3_1_sino)
    conv3_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv3_1l_sino)
    conv3_2l_sino = LeakyReLU(alpha=0.2)(conv3_2_sino)            
    conv3_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv3_2l_sino)
    conv3_3l_sino = LeakyReLU(alpha=0.2)(conv3_3_sino)  
    conv3_dr_sino = SpatialDropout2D(0.1)(conv3_3l_sino)

    conv4_1_sino = Conv2D(64, 3, strides=2, padding = pad,
            kernel_initializer = 'he_normal')(conv3_dr_sino)
    conv4_1l_sino = LeakyReLU(alpha=0.2)(conv4_1_sino)
    conv4_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv4_1l_sino)
    conv4_2l_sino = LeakyReLU(alpha=0.2)(conv4_2_sino)            
    conv4_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv4_2l_sino)
    conv4_3l_sino = LeakyReLU(alpha=0.2)(conv4_3_sino)  
    conv4_dr_sino = SpatialDropout2D(0.1)(conv4_3l_sino)

    conv5_1_sino = Conv2D(64, 3, strides=2, padding = pad,
            kernel_initializer = 'he_normal')(conv4_dr_sino)
    conv5_1l_sino = LeakyReLU(alpha=0.2)(conv5_1_sino)
    conv5_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv5_1l_sino)
    conv5_2l_sino = LeakyReLU(alpha=0.2)(conv5_2_sino)            
    conv5_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv5_2l_sino)
    conv5_3l_sino = LeakyReLU(alpha=0.2)(conv5_3_sino)  
    conv5_dr_sino = SpatialDropout2D(0.1)(conv5_3l_sino)

    flat_sino = Flatten()(conv5_dr_sino)

    den1_sino = Dense(500, kernel_initializer='random_normal')(flat_sino)
    den1_l_sino = LeakyReLU(alpha=0.2)(den1_sino)
    den1_dr_sino = Dropout(0.1)(den1_l_sino)
    
    den2_sino = Dense(500, kernel_initializer='random_normal')(den1_dr_sino)
    den2_l_sino = LeakyReLU(alpha=0.2)(den2_sino)
    den2_dr_sino = Dropout(0.1)(den2_l_sino)

    den3_sino = Dense(500, kernel_initializer='random_normal')(den2_dr_sino)
    den3_l_sino = LeakyReLU(alpha=0.2)(den3_sino)
    den3_dr_sino = Dropout(0.1)(den3_l_sino)

    den_sino = Dense(int(np.ceil(up_6_pad_sino / 2)) * int(np.ceil(up_6_pad_sino / 2)) * 8, kernel_initializer='random_normal', activation='linear')(den3_dr_sino)
    den_dr_sino = Dropout(0.1)(den_sino)

    resh_sino = Reshape((int(np.ceil(up_6_pad_sino / 2)), int(np.ceil(up_6_pad_sino / 2)), 8))(den_dr_sino)

    conv6_1_sino = Conv2D(64, 3, padding = pad,
            kernel_initializer = 'he_normal')(resh_sino)
    conv6_1l_sino = LeakyReLU(alpha=0.2)(conv6_1_sino)
    conv6_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv6_1l_sino)
    conv6_2l_sino = LeakyReLU(alpha=0.2)(conv6_2_sino)            
    conv6_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv6_2l_sino)
    conv6_3l_sino = LeakyReLU(alpha=0.2)(conv6_3_sino)  
    conv6_dr_sino = SpatialDropout2D(0.1)(conv6_3l_sino)

    up6_sino = Conv2D(64, 2, padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_dr_sino))
    up6_l_sino = LeakyReLU(alpha=0.2)(up6_sino) 

    if up_6_pad_sino % 2 != 0:
        up6_l_sino = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(up6_l_sino)

    conv7_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(up6_l_sino)
    conv7_2l_sino = LeakyReLU(alpha=0.2)(conv7_2_sino)            
    conv7_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv7_2l_sino)
    conv7_3l_sino = LeakyReLU(alpha=0.2)(conv7_3_sino)  
    conv7_dr_sino = SpatialDropout2D(0.1)(conv7_3l_sino)

    up7_sino = Conv2D(64, 2, padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_dr_sino))
    up7_l_sino = LeakyReLU(alpha=0.2)(up7_sino) 

    if up_7_pad_sino % 2 != 0:
        up7_l_sino = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(up7_l_sino)

    conv8_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(up7_l_sino)
    conv8_2l_sino = LeakyReLU(alpha=0.2)(conv8_2_sino)            
    conv8_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv8_2l_sino)
    conv8_3l_sino = LeakyReLU(alpha=0.2)(conv8_3_sino)  
    conv8_dr_sino = SpatialDropout2D(0.1)(conv8_3l_sino)

    up8_sino = Conv2D(64, 2, padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_dr_sino))
    up8_l_sino = LeakyReLU(alpha=0.2)(up8_sino) 

    if up_8_pad_sino % 2 != 0:
        up8_l_sino = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(up8_l_sino)

    conv9_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(up8_l_sino)
    conv9_2l_sino = LeakyReLU(alpha=0.2)(conv9_2_sino)            
    conv9_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv9_2l_sino)
    conv9_3l_sino = LeakyReLU(alpha=0.2)(conv9_3_sino)  
    conv9_dr_sino = SpatialDropout2D(0.1)(conv9_3l_sino)

    up9_sino = Conv2D(64, 2, padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9_dr_sino))
    up9_l_sino = LeakyReLU(alpha=0.2)(up9_sino) 

    if up_9_pad_sino % 2 != 0:
        up9_l_sino = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(up9_l_sino)

    conv10_2_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(up9_l_sino)
    conv10_2l_sino = LeakyReLU(alpha=0.2)(conv10_2_sino)            
    conv10_3_sino = Conv2D(64, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv10_2l_sino)
    conv10_3l_sino = LeakyReLU(alpha=0.2)(conv10_3_sino)  
    conv10_4_sino = Conv2D(32, 3, padding = pad, 
            kernel_initializer = 'he_normal')(conv10_3l_sino)
    conv10_4l_sino = LeakyReLU(alpha=0.2)(conv10_4_sino)      
    conv10_dr_sino = SpatialDropout2D(0.1)(conv10_4l_sino)

    conv11_sino = Conv2D(1, 1, activation = 'linear')(conv10_dr_sino)

    model = Model(sinogram_in, conv11_sino)

    return model


def unet2D_large(ntr, pad='same'):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, ntr, 1))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(lowres_in)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)

    conv2 = Conv2D(128, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = SpatialDropout2D(0.1)(conv2)

    conv3 = Conv2D(256, 3, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = SpatialDropout2D(0.1)(conv3)

    conv4 = Conv2D(512, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = SpatialDropout2D(0.1)(conv4)

    conv5 = Conv2D(512, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = SpatialDropout2D(0.1)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up6)

    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = SpatialDropout2D(0.1)(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up7)

    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = SpatialDropout2D(0.1)(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = SpatialDropout2D(0.1)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up9)

    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = SpatialDropout2D(0.1)(conv9)

    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    model = Model(lowres_in, conv10)

    return model


def unet2D_small_largekernel(ntr, pad='same'):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, ntr, 1))
    
    conv1 = Conv2D(64, 5, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal')(lowres_in)
    conv1 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)

    conv2 = Conv2D(64, 5, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv2 = SpatialDropout2D(0.1)(conv2)

    conv3 = Conv2D(64, 5, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv3 = SpatialDropout2D(0.1)(conv3)

    conv4 = Conv2D(64, 5, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv3)
    conv4 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv4 = SpatialDropout2D(0.1)(conv4)

    conv5 = Conv2D(64, 5, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv4)
    conv5 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv5)
    conv5 = SpatialDropout2D(0.1)(conv5)

    up6 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up6)

    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv6)
    conv6 = SpatialDropout2D(0.1)(conv6)

    up7 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up7)

    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv7)
    conv7 = SpatialDropout2D(0.1)(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv8)
    conv8 = SpatialDropout2D(0.1)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up9)

    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(32, 5, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal')(conv9)
    conv9 = SpatialDropout2D(0.1)(conv9)

    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    model = Model(lowres_in, conv10)

    return model

def unet2D_small_weightnorm(ntr, pad='same'):

    up_9_pad = ntr

    up_8_pad = np.ceil(up_9_pad / 2)
    up_7_pad = np.ceil(up_8_pad / 2)
    up_6_pad = np.ceil(up_7_pad / 2)
    print(up_9_pad, up_8_pad, up_7_pad, up_6_pad)

    lowres_in = Input(shape=(ntr, ntr, 1))
    
    conv1 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad,
            kernel_initializer = 'he_normal'))(lowres_in)
    conv1 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv1)
    conv1 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)

    conv2 = WeightNormalization(Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv1)
    conv2 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv2)
    conv2 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv2)
    conv2 = SpatialDropout2D(0.1)(conv2)

    conv3 = WeightNormalization(Conv2D(64, 3, strides = 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv2)
    conv3 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv3)
    conv3 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv3)
    conv3 = SpatialDropout2D(0.1)(conv3)

    conv4 = WeightNormalization(Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv3)
    conv4 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv4)
    conv4 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv4)
    conv4 = SpatialDropout2D(0.1)(conv4)

    conv5 = WeightNormalization(Conv2D(64, 3, strides=2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv4)
    conv5 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv5)
    conv5 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv5)
    conv5 = SpatialDropout2D(0.1)(conv5)

    up6 = WeightNormalization(Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(UpSampling2D(size = (2,2))(conv5))

    if up_6_pad % 2 != 0:
        up6 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up6)

    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(merge6)
    conv6 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv6)
    conv6 = SpatialDropout2D(0.1)(conv6)

    up7 = WeightNormalization(Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(UpSampling2D(size = (2,2))(conv6))

    if up_7_pad % 2 != 0:
        up7 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up7)

    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(merge7)
    conv7 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv7)
    conv7 = SpatialDropout2D(0.1)(conv7)

    up8 = WeightNormalization(Conv2D(64, 2, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(UpSampling2D(size = (2,2))(conv7))

    if up_8_pad % 2 != 0:
        up8 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up8)

    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(merge8)
    conv8 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv8)
    conv8 = SpatialDropout2D(0.1)(conv8)

    up9 = WeightNormalization(Conv2D(64, 2, activation = 'relu', padding = pad, 
        kernel_initializer = 'he_normal'))(UpSampling2D(size = (2,2))(conv8))

    if up_9_pad % 2 != 0:
        up9 = tf.keras.layers.Cropping2D(
                cropping=((1, 0), (1, 0)), data_format=None)(up9)

    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(merge9)
    conv9 = WeightNormalization(Conv2D(64, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv9)
    conv9 = WeightNormalization(Conv2D(32, 3, activation = 'relu', padding = pad, 
            kernel_initializer = 'he_normal'))(conv9)
    conv9 = SpatialDropout2D(0.1)(conv9)

    conv10 = WeightNormalization(Conv2D(1, 1, activation = 'linear'))(conv9)

    model = Model(lowres_in, conv10)

    return model

