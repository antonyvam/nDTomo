# -*- coding: utf-8 -*-
"""
Metrics for data science

@author: Antony Vamvakeros
"""

from numpy import sum, square, mean

def tv_image(im, method = 'isotropic'):
    
    '''
    Calculate and return the anisotropic (2D) total variation of an image. The total variation is the sum of the absolute differences 
    for neighboring pixel-values in the input images. This measures how much noise is in the images. 
    This can be used as a loss-function during optimization so as to suppress noise in images.
    
    Adapted from tensorflow
    '''
    
    pixel_dif1 = im[1:, :] - im[:-1, :]
    pixel_dif2 = im[:, 1:] - im[:, :-1]
    
    if method == 'isotropic':
        
        tv = sum(pixel_dif1) + sum(pixel_dif2)
   
    
    elif method == 'anisotropic':
        
        tv = sum(square(pixel_dif1**2 + pixel_dif2**2))
       
    return(tv)

def ssim_images(x, y, max_val, compensation=1.0, k1=0.01, k2=0.03):
    
    '''

    x: First image
    y: Second image
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).    
    
    Adapted from tensorflow
    '''
    
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
      
    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = mean(x)
    mean1 = mean(y)
    num0 = mean0 * mean1 * 2.0
    den0 = square(mean0) + square(mean1)
    luminance = (num0 + c1) / (den0 + c1)
      
    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = mean(x * y) * 2.0
    den1 = mean(square(x) + square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
      
    # SSIM score is the product of the luminance and contrast-structure measures.
    
    ssim_val = mean(luminance * cs)
    
    return(ssim_val)