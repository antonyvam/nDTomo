# -*- coding: utf-8 -*-
"""
Filters for 2D and 3D data

@author: Antony Vamvakeros
"""

from numpy import mean
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet, denoise_nl_means, estimate_sigma

def filter_gaussian(data, sigma = 1):
    """
    Applies a Gaussian filter to the data.
    
    Args:
    data : ndarray
        Input data to filter.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard 
        deviations of the Gaussian filter are given for each axis as a 
        sequence, or as a single number, in which case it is equal for all axes.
        
    Returns:
    out : ndarray
        Filtered data.
    """    
    return(gaussian_filter(data, sigma=sigma))

def filter_median(data, size = 3):
    """
    Applies a median filter to the data.
    
    Args:
    data : ndarray
        Input data to filter.
    size : scalar or sequence of scalars, optional
        The sizes of the neighborhood over which the median is computed. 
        The size must be odd and greater than or equal to 3 in all dimensions.
        
    Returns:
    out : ndarray
        Filtered data.
    """    
    return(median_filter(data, size=size))

def filter_bilateral(data, sigma_color=0.05, sigma_spatial=15):
    """
    Applies a bilateral filter to the data. This filter is primarily designed for 2D data.
    
    Args:
    data : ndarray
        Input data to filter.
    sigma_color : float
        The range within which two colors in the image are considered close.
    sigma_spatial : float
        A range within which two pixels are considered close in their spatial coordinates.
        
    Returns:
    out : ndarray
        Filtered data.
    """
    return(denoise_bilateral(data, sigma_color=sigma_color, sigma_spatial=sigma_spatial))

def filter_tv(data, weight=0.1):
    """
    Applies a total variation (TV) denoising filter to the data.
    
    Args:
    data : ndarray
        Input data to filter.
    weight : float
        Denoising weight. The greater weight, the more denoising (at the expense of fidelity to data).
        
    Returns:
    out : ndarray
        Filtered data.
    """    
    return(denoise_tv_chambolle(data, weight=weight))


def filter_wavelet(data, multichannel=False):
    """
    Applies a wavelet denoising filter to the data.
    
    Args:
    data : ndarray
        Input data to filter.
    multichannel : bool, optional (default: False)
        Whether the last axis of the image is to be interpreted as multiple channels.
        
    Returns:
    out : ndarray
        Filtered data.
    """    
    return(denoise_wavelet(data, multichannel=multichannel))

def filter_nonlocalmeans(data, multichannel=False, fast_mode=True, patch_size=5, patch_distance=3):
    """
    Applies a non-local means denoising filter to the data.
    
    Args:
    data : ndarray
        Input data to filter.
    multichannel : bool
        Whether the image is a color image (default is grayscale).
    fast_mode : bool
        If True, a fast version is used at the expense of performance.
    patch_size : int
        Size of patches used for denoising.
    patch_distance : int
        Maximal distance where to search patches used for denoising.
        
    Returns:
    out : ndarray
        Filtered data.
    """
    sigma_estimated = mean(estimate_sigma(data, multichannel=multichannel))
    return(denoise_nl_means(data, h=1.15 * sigma_estimated, fast_mode=fast_mode, 
                                   patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel))
    