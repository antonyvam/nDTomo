# -*- coding: utf-8 -*-
"""
Various tensorflow functions

@author: Antony Vamvakeros
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from numpy import less, greater, Inf, zeros_like
from tqdm import tqdm
from tensorflow_addons.image import rotate
from nDTomo.nn.tomo_tf import tf_tomo_transf

def tf_gpu_devices():
        
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")

def tf_gpu_allocateRAM():
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def rotate_tf(im, ang):
    
    '''
    Rotate a 2D image using tensorflow
    '''
    
    im = rotate(tf_tomo_transf(im), ang, interpolation = 'bilinear')[0,:,:,0]

    return(im)

def rotate_xz(vol, ang):
    
    
    dims = vol.shape
    voln = zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[1])):    
    
        voln[:,ii,:] = rotate(vol[:,ii,:].reshape(dims[0], dims[2], 1), ang, interpolation = 'bilinear')[:,:,0]
    
    return(voln)

def rotate_xy(vol, ang):
    
    dims = vol.shape
    voln = zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[2])):    
    
        voln[:,:,ii] = rotate(vol[:,:,ii].reshape(dims[0], dims[1], 1), ang, interpolation = 'bilinear')[:,:,0]
    
    return(voln)

def rotate_yz(vol, ang):
    
    dims = vol.shape
    voln = zeros_like(vol)
    
    for ii in tqdm(range(voln.shape[0])):    
    
        voln[ii,:,:] = rotate(vol[ii,:,:].reshape(dims[1], dims[2], 1), ang, interpolation = 'bilinear')[:,:,0]
    
    return(voln)

def extract_patches(x, PATCH_WIDTH, PATCH_HEIGHT):
    '''
    Edited from: https://gist.github.com/hwaxxer/17ea565f86b748ba9471546b2532d0cf

    ksizes is used to decide the dimensions of each patch, or in other words, how many pixels each patch should contain.

    strides denotes the length of the gap between the start of one patch and the start of the next consecutive patch within the original image.

    rates is a number that essentially means our patch should jump by rates pixels in the original image for each consecutive pixel that ends up in our patch. (The example below helps illustrate this.)

    padding is either "VALID", which means every patch must be fully contained in the image, or "SAME", which means patches are allowed to be incomplete (the remaining pixels will be filled in with zeroes).

    '''
    
    ksizes = [1, PATCH_WIDTH, PATCH_HEIGHT, 1]
    strides = [1, PATCH_WIDTH, PATCH_HEIGHT, 1]
    rates = [1, 1, 1, 1]
    padding = 'SAME'
    return tf.image.extract_patches(x, ksizes, strides, rates, padding)

def extract_patches_inverse(x, y, tape, PATCH_WIDTH, PATCH_HEIGHT):
    '''
    Edited from: https://gist.github.com/hwaxxer/17ea565f86b748ba9471546b2532d0cf
    '''    
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, PATCH_WIDTH, PATCH_HEIGHT)
    grad = tape.gradient(_y, _x)
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tape.gradient(_y, _x, output_gradients=y) / grad

def merge_patches(img, patches, PATCH_WIDTH, PATCH_HEIGHT):
    '''
    Edited from: https://gist.github.com/hwaxxer/17ea565f86b748ba9471546b2532d0cf
    '''    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img)
        inv = extract_patches_inverse(img, patches, tape, PATCH_WIDTH, PATCH_HEIGHT)
        
    return(inv)
	

def image_to_patches(image, patch_width, patch_height):

    '''
    Edited from: https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches    
    '''
    
    image_height = image.shape[1]
    image_width = image.shape[2]
    
    height = int(tf.math.ceil(image_height/patch_height)*patch_height)
    width = int(tf.math.ceil(image_width/patch_width)*patch_width)

    image_resized = tf.squeeze(tf.image.resize_with_crop_or_pad(image, height, width))
    image_reshaped = tf.reshape(image_resized, [height // patch_height, patch_height, -1, patch_width])
    image_transposed = tf.transpose(image_reshaped, [0, 2, 1, 3])
    return tf.reshape(image_transposed, [-1, patch_height, patch_width, 1])


def patches_to_image(patches, image_height, image_width, patch_height, patch_width):

    '''
    Edited from: https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches    
    '''

    height = int(tf.math.ceil(image_height/patch_height)*patch_height)
    width = int(tf.math.ceil(image_width/patch_width)*patch_width)

    image_reshaped = tf.reshape(tf.squeeze(patches), [height // patch_height, width // patch_width, patch_height, patch_width])
    image_transposed = tf.transpose(image_reshaped, [0, 2, 1, 3])
    image_resized = tf.reshape(image_transposed, [height, width, 1])
    return tf.image.resize_with_crop_or_pad(image_resized, image_height, image_width)

class ReduceLROnPlateau_custom(Callback):

    '''
    Custom reduce learning rate on plateau callback, it can be used in custom training loops
    '''
    
    def __init__(self,
                  ## Custom modification:  Deprecated due to focusing on validation loss
                  # monitor='val_loss',
                  factor=0.5,
                  patience=10,
                  verbose=0,
                  mode='auto',
                  min_delta=1e-4,
                  cooldown=0,
                  min_lr=0,
                  sign_number = 4,
                  ## Custom modification: Passing optimizer as arguement
                  optim_lr = None,
                  ## Custom modification:  linearly reduction learning
                  reduce_lin = False,
                  **kwargs):
    
        ## Custom modification:  Deprecated
        # super(ReduceLROnPlateau, self).__init__()
    
        ## Custom modification:  Deprecated
        # self.monitor = monitor
        
        ## Custom modification: Optimizer Error Handling
        if tf.is_tensor(optim_lr) == False:
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        ## Custom modification: Passing optimizer as arguement
        self.optim_lr = optim_lr  
    
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.sign_number = sign_number
        
    
        ## Custom modification: linearly reducing learning
        self.reduce_lin = reduce_lin
        self.reduce_lr = True
        
    
        self._reset()

    def _reset(self):
         """Resets wait counter and cooldown counter.
         """
         if self.mode not in ['auto', 'min', 'max']:
             print('Learning Rate Plateau Reducing mode %s is unknown, '
                             'fallback to auto mode.', self.mode)
             self.mode = 'auto'
         if (self.mode == 'min' or
                 ## Custom modification: Deprecated due to focusing on validation loss
                 # (self.mode == 'auto' and 'acc' not in self.monitor)):
                 (self.mode == 'auto')):
             self.monitor_op = lambda a, b: less(a, b - self.min_delta)
             self.best = Inf
         else:
             self.monitor_op = lambda a, b: greater(a, b + self.min_delta)
             self.best = -Inf
         self.cooldown_counter = 0
         self.wait = 0

    def on_train_begin(self, logs=None):
      self._reset()
    
    def on_epoch_end(self, epoch, loss, logs=None):
    
    
        logs = logs or {}
        ## Custom modification: Optimizer
        # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
        # and therefore can be modified to          
        logs['lr'] = float(self.optim_lr.numpy())
    
        ## Custom modification: Deprecated due to focusing on validation loss
        # current = logs.get(self.monitor)
    
        current = float(loss)
    
        ## Custom modification: Deprecated due to focusing on validation loss
        # if current is None:
        #     print('Reduce LR on plateau conditioned on metric `%s` '
        #                     'which is not available. Available metrics are: %s',
        #                     self.monitor, ','.join(list(logs.keys())))
    
        # else:
    
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
    
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
    
                ## Custom modification: Optimizer Learning Rate
                # old_lr = float(K.get_value(self.model.optimizer.lr))
                old_lr = float(self.optim_lr.numpy())
                if old_lr > self.min_lr and self.reduce_lr == True:
                    ## Custom modification: Linear learning Rate
                    if self.reduce_lin == True:
                        new_lr = old_lr * self.factor
                        ## Custom modification: Error Handling when learning rate is below zero
                        if new_lr <= 0:
                            print('Learning Rate is below zero: {}, '
                            'fallback to minimal learning rate: {}. '
                            'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))  
                            self.reduce_lr = False                           
                    else:
                        new_lr = old_lr * self.factor                   
    
    
                    new_lr = max(new_lr, self.min_lr)
    
    
                    ## Custom modification: Optimizer Learning Rate
                    # K.set_value(self.model.optimizer.lr, new_lr)
                    self.optim_lr.assign(new_lr)
    
                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                'rate to %s.' % (epoch + 1, float(new_lr)))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
    
    def in_cooldown(self):
      return self.cooldown_counter > 0