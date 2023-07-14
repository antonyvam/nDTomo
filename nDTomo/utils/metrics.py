# -*- coding: utf-8 -*-
"""
Metrics for data science

@author: Antony Vamvakeros
"""

from skimage.metrics import structural_similarity as ssim
from numpy import sum, mean, max, log10, sqrt, diff, abs


def mse(data1, data2):
    """
    Computes the Mean Squared Error (MSE) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The mean squared error value.
    """    
    return mean((data1 - data2) ** 2)


def psnr(data1, data2):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The PSNR value.
    """    
    mse_val = mse(data1, data2)
    if mse_val == 0:  # Same volumes
        return 100
    max_pixel = max(data2)
    return 20 * log10(max_pixel / sqrt(mse_val))

def compute_ssim(data1, data2):
    """
    Computes the Structural Similarity Index (SSIM) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The SSIM value.
    """    
    return ssim(data1, data2, data_range=data2.max() - data2.min())


def normalized_cross_correlation(data1, data2):
    """
    Computes the Normalized Cross-Correlation (NCC) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The NCC value.
    """    
    numerator = sum((data1 - mean(data1)) * (data2 - mean(data2)))
    denominator = sqrt(sum((data1 - mean(data1))**2) * sum((data2 - mean(data2))**2))
    return numerator/denominator


def calculate_total_variation(image, tv_type):
    """
    Calculates the isotropic or anisotropic total variation of an image.

    Parameters:
        image (numpy.ndarray): Input image as a 2D NumPy array.
        tv_type (str): Type of total variation to calculate: 'isotropic' or 'anisotropic'.

    Returns:
        float: Total variation value.
    """
    # Calculate differences in x and y directions
    dx = diff(image, axis=1)
    dy = diff(image, axis=0)

    if tv_type == 'isotropic':
        # Calculate isotropic total variation
        tv = sum(sqrt(dx**2 + dy**2))
    elif tv_type == 'anisotropic':
        # Calculate anisotropic total variation
        tv = sum(abs(dx)) + sum(abs(dy))
    else:
        raise ValueError("Invalid TV type. Allowed values are 'isotropic' and 'anisotropic'.")

    return tv










