# -*- coding: utf-8 -*-
"""
Custom layers for Tensorflow

@author: Antony Vamvakeros
"""

from tensorflow.keras import layers
from tensorflow import matmul, multiply, reduce_mean
from tensorflow.nn import softmax, sigmoid

class SelfAttention(layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, 1)
        self.key_conv = layers.Conv2D(channels // 8, 1)
        self.value_conv = layers.Conv2D(channels, 1)
        self.gamma = layers.LayerVariable(0)

    def attention(self, query, key, value):
        score = matmul(query, key, transpose_b=True)
        weight = softmax(score)
        return matmul(weight, value)

    def call(self, inputs):
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        attention = self.attention(query, key, value)
        attention = layers.Reshape((inputs.shape[1], inputs.shape[2], self.channels))(attention)
        attention = layers.Multiply()([inputs, attention])
        return attention

class SpatialAttention2D(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention2D, self).__init__(**kwargs)
        self.conv1x1 = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu')
    
    def call(self, inputs):
        # reduce dimensionality of input feature maps
        attention_maps = self.conv1x1(inputs)
        # apply softmax to create attention maps
        attention_maps = softmax(attention_maps, axis=-1)
        # apply attention maps to input feature maps
        attention_output = multiply(inputs, attention_maps)
        return attention_output

class ScaleAwareAttention2D(layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleAwareAttention2D, self).__init__(**kwargs)

    def call(self, inputs):
        # apply global average pooling to input feature maps
        global_average_pooling = reduce_mean(inputs, axis=[1, 2])
        # apply softmax to create attention weights
        attention_weights = softmax(global_average_pooling, axis=-1)
        # apply attention weights to input feature maps
        attention_output = multiply(inputs, attention_weights)
        return attention_output


