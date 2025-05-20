# -*- coding: utf-8 -*-
"""
Metrics for data science

@author: Antony Vamvakeros
"""

from skimage.metrics import structural_similarity as ssim
import numpy as np

def mae(data1, data2):
    """
    Computes the Mean Absolute Error (MAE) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The mean squared error value.
    """    
    return np.mean(abs(data1 - data2))

def mse(data1, data2):
    """
    Computes the Mean Squared Error (MSE) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The mean squared error value.
    """    
    return np.mean((data1 - data2) ** 2)


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
    max_pixel = max(data1)
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def ssim_data(data1, data2):
    """
    Computes the Structural Similarity Index (SSIM) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The SSIM value.
    """    
    data_range = np.max((data2.max(), data1.max())) - np.min((data2.min(), data1.min()))
    return ssim(data1, data2, data_range=data_range)


def normalized_cross_correlation(data1, data2):
    """
    Computes the Normalized Cross-Correlation (NCC) between two arrays.

    Parameters:
        data1 (numpy.ndarray): First input array.
        data2 (numpy.ndarray): Second input array.

    Returns:
        float: The NCC value.
    """    
    numerator = np.sum((data1 - np.mean(data1)) * (data2 - np.mean(data2)))
    denominator = np.sqrt(np.sum((data1 - np.mean(data1))**2) * np.sum((data2 - np.mean(data2))**2))
    return numerator/denominator


def total_variation_image(image, tv_type):
    """
    Calculates the isotropic or anisotropic total variation of an image.

    Parameters:
        image (numpy.ndarray): Input image as a 2D NumPy array.
        tv_type (str): Type of total variation to calculate: 'isotropic' or 'anisotropic'.

    Returns:
        float: Total variation value.
    """
    # Calculate differences in x and y directions
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    if tv_type == 'isotropic':
        # Calculate isotropic total variation
        tv = np.sum(np.sqrt(dx**2 + dy**2))
    elif tv_type == 'anisotropic':
        # Calculate anisotropic total variation
        tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
    else:
        raise ValueError("Invalid TV type. Allowed values are 'isotropic' and 'anisotropic'.")

    return tv


def total_variation_volume(volume, tv_type):
    """
    Calculates the isotropic or anisotropic total variation of a volume.

    Parameters:
        volume (numpy.ndarray): Input volume as a 3D NumPy array.
        tv_type (str): Type of total variation to calculate: 'isotropic' or 'anisotropic'.

    Returns:
        float: Total variation value.
    """
    # Calculate differences in x, y, and z directions
    dx = np.diff(volume, axis=2)
    dy = np.diff(volume, axis=1)
    dz = np.diff(volume, axis=0)

    if tv_type == 'isotropic':
        # Calculate isotropic total variation
        tv = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    elif tv_type == 'anisotropic':
        # Calculate anisotropic total variation
        tv = np.sum(np.abs(dx)) + np.sum(np.abs(dy)) + np.sum(np.abs(dz))
    else:
        raise ValueError("Invalid TV type. Allowed values are 'isotropic' and 'anisotropic'.")

    return tv


def calculate_rmse(data1, data2):
    """
    Calculates the root mean squared error (RMSE) between two datasets.

    Parameters:
        volume1 (numpy.ndarray): First input volume as a NumPy array.
        volume2 (numpy.ndarray): Second input volume as a NumPy array.

    Returns:
        float: Root mean squared error value.
    """
    # Compute the squared difference between the volumes
    squared_diff = (data1 - data2) ** 2

    # Compute the mean squared error
    mse = np.mean(squared_diff)

    # Compute the root mean squared error
    rmse = np.sqrt(mse)

    return rmse

def Rwp(y_obs, y_calc, weights=None):
    
    """
    Function to compute the weighted profile R-factor (Rwp) used in X-ray diffraction analysis.
    
    Parameters:
    y_obs (numpy array): The observed (experimental) data points.
    y_calc (numpy array): The calculated (or model) data points.
    weights (numpy array, optional): The weights for each data point. Defaults to 1/y_obs.
    
    Returns:
    float: The calculated Rwp value.
    """
    
    if weights is None:
        weights = 1 / y_obs
    
    numerator = sum(weights * (y_obs - y_calc)**2)
    denominator = sum(weights * y_obs**2)
    
    return np.sqrt(numerator / denominator)


def Rexp(y_obs, y_calc, weights=None):
    """
    Function to compute the expected profile R-factor (Rexp) used in X-ray diffraction analysis.
    
    Parameters:
    y_obs (numpy array): The observed (experimental) data points.
    y_calc (numpy array): The calculated (or model) data points.
    weights (numpy array, optional): The weights for each data point. Defaults to 1/y_obs.
    
    Returns:
    float: The calculated Rexp value.
    """
    
    if weights is None:
        weights = 1 / y_obs

    denominator = np.sum(weights * y_obs**2)
    
    return np.sqrt(1 / denominator)


def chi_square(y_obs, y_calc, weights=None):
    """
    Function to compute the chi-square value used in statistical analysis.
    
    Parameters:
    y_obs (numpy array): The observed (experimental) data points.
    y_calc (numpy array): The calculated (or model) data points.
    weights (numpy array, optional): The weights for each data point. Defaults to 1/y_obs.
    
    Returns:
    float: The calculated chi-square value.
    """
    
    if weights is None:
        weights = 1 / y_obs
    
    return np.sum(weights * (y_obs - y_calc)**2)


def compute_goodness_of_fit(y_obs, y_calc, weights=None):
    """
    Function to compute the goodness of fit used in statistical analysis.
    
    Parameters:
    y_obs (numpy array): The observed (experimental) data points.
    y_calc (numpy array): The calculated (or model) data points.
    weights (numpy array, optional): The weights for each data point. Defaults to 1/y_obs.
    
    Returns:
    float: The calculated goodness of fit value.
    """
    
    return Rwp(y_obs, y_calc, weights)**2 / Rexp(y_obs, y_calc, weights)