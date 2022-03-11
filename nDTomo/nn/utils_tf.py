# -*- coding: utf-8 -*-
"""
Various tensorflow functions

@author: Antony Vamvakeros
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from numpy import less, greater, Inf

def tf_gpu_devices():
        
    if tf.test.gpu_device_name(): 
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")

def tf_gpu_allocateRAM():
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
	
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