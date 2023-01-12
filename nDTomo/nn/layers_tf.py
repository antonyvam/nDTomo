# -*- coding: utf-8 -*-
"""
Custom layers for Tensorflow

@author: Antony Vamvakeros
"""

from tensorflow.keras import layers
from tensorflow import matmul
from tensorflow.nn import softmax

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

    