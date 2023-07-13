# -*- coding: utf-8 -*-
"""
Filters for 2D and 3D data

@author: Antony Vamvakeros
"""

from numpy import mean
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet, denoise_nl_means, estimate_sigma

def filter_gaussian(data, sigma = 1):
    return(gaussian_filter(data, sigma=sigma))

def filter_median(data, size = 1):
    return(median_filter(data, size=3))

def filter_bilateral(data, sigma_color=0.05, sigma_spatial=15):
    '''sigma_color is the range within which two colors in the image are considered close, 
    and sigma_spatial is a range within which two pixels are considered close in their spatial coordinates.''' 
    return(denoise_bilateral(data, sigma_color=sigma_color, sigma_spatial=sigma_spatial))

def filter_tv(data, weight=0.1):
    return(denoise_tv_chambolle(data, weight=weight))


def filter_wavelet(data, multichannel=False):
    return(denoise_wavelet(data, multichannel=multichannel))

def filter_nonlocalmeans(data, multichannel=False, fast_mode=True, patch_size=5, patch_distance=3):
    
    '''patch_size determines the size of patches to compare for non-local means, 
    and patch_distance is the max distance where to search patches used for the non-local means. 
    multichannel should be set to False for grayscale images, and True for color images. 
    The fast_mode argument trades off speed against performance, and if it is True, then the computation time can be significantly reduced.'''
    
    sigma_estimated = mean(estimate_sigma(data, multichannel=multichannel))
    return(denoise_nl_means(data, h=1.15 * sigma_estimated, fast_mode=fast_mode, 
                                   patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel))
    