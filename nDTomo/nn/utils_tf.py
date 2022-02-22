# -*- coding: utf-8 -*-
"""
Various tensorflow functions

@author: Antony Vamvakeros
"""

import tensorflow as tf

def tf_gpu_devices():
        
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")

