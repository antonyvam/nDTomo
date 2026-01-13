# -*- coding: utf-8 -*-
"""
Models and functions for peak fitting

@author: Dr A. Vamvakeros
"""

import numpy as np
import h5py
from tqdm import tqdm
from scipy.optimize import curve_fit

tand = lambda x: np.tan(x*np.pi/180.)
atand = lambda x: 180.*np.arctan(x)/np.pi

def Linear(x, a=0, b=0):
    """
    Computes a linear function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the linear function.
    a : float, optional, default=0
        Slope of the line.
    b : float, optional, default=0
        Intercept of the line.

    Returns
    -------
    numpy.ndarray or float
        Output of the linear function.
    """
    return a*x+b

def Quadratic(x, a=0, b=0, c=0):
    """
    Computes a quadratic function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the quadratic function.
    a : float, optional, default=0
        Coefficient for the quadratic term.
    b : float, optional, default=0
        Coefficient for the linear term.
    c : float, optional, default=0
        Constant term.

    Returns
    -------
    numpy.ndarray or float
        Output of the quadratic function.
    """
    return a*x**2 + b*x + c

def Gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """
    Computes a Gaussian function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the Gaussian function.
    amplitude : float, optional, default=1.0
        Amplitude of the Gaussian peak.
    center : float, optional, default=0.0
        Center of the Gaussian peak.
    sigma : float, optional, default=1.0
        Standard deviation of the Gaussian peak.

    Returns
    -------
    numpy.ndarray or float
        Output of the Gaussian function.
    """
    return (2.0*np.sqrt(np.log(2.0)/np.pi))*(amplitude/sigma ) * np.exp(-(4.0*np.log(2.0)/sigma**2) * (x - center)**2)
    
def Lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """
    Computes a Lorentzian function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the Lorentzian function.
    amplitude : float, optional, default=1.0
        Amplitude of the Lorentzian peak.
    center : float, optional, default=0.0
        Center of the Lorentzian peak.
    sigma : float, optional, default=1.0
        Width parameter of the Lorentzian peak.

    Returns
    -------
    numpy.ndarray or float
        Output of the Lorentzian function.
    """
    return (2.0 / np.pi) * (amplitude / sigma) /  (1 + (4.0 / sigma**2) * (x - center**2))

def G(x, center=0.0, sigma=1.0):
    """
    Computes a normalized Gaussian function.

    This function is a special case of the Gaussian function with a default 
    amplitude of 1.0.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the Gaussian function.
    center : float, optional, default=0.0
        Center of the Gaussian peak.
    sigma : float, optional, default=1.0
        Standard deviation of the Gaussian peak.

    Returns
    -------
    numpy.ndarray or float
        Output of the normalized Gaussian function.
    """
    return (2.0*np.sqrt(np.log(2.0)/np.pi))*(1/sigma ) * np.exp(-(4.0*np.log(2.0)/sigma**2) * (x - center)**2)    

def L(x, center=0.0, sigma=1.0):
    """
    Computes a normalized Lorentzian function.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the Lorentzian function.
    center : float, optional, default=0.0
        Center of the Lorentzian peak.
    sigma : float, optional, default=1.0
        Width parameter of the Lorentzian peak.

    Returns
    -------
    numpy.ndarray or float
        Output of the normalized Lorentzian function.
    """
    return (2.0 / np.pi) * (1 / sigma) /  (1 + (4.0 / sigma**2) * (x - center**2))

def PVoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """
    Computes a pseudo-Voigt function as a weighted sum of Gaussian and Lorentzian functions.

    Parameters
    ----------
    x : numpy.ndarray or float
        Input value(s) for the pseudo-Voigt function.
    amplitude : float, optional, default=1.0
        Amplitude of the pseudo-Voigt function.
    center : float, optional, default=0.0
        Center of the peak.
    sigma : float, optional, default=1.0
        Width parameter shared by both Gaussian and Lorentzian components.
    fraction : float, optional, default=0.5
        Fractional contribution of the Lorentzian component. The Gaussian contribution 
        is `(1 - fraction)`.

    Returns
    -------
    numpy.ndarray or float
        Output of the pseudo-Voigt function.
    """
    return amplitude * ((1-fraction) * G(x, center, sigma) + fraction * L(x, center, sigma))

def CombGaussians(x, pars):
    """
    Computes a combination of multiple Gaussian functions with a linear background.

    Parameters
    ----------
    x : numpy.ndarray
        Input value(s) for the Gaussian functions.
    pars : list or numpy.ndarray
        Parameter vector for the Gaussian functions and linear background:
        - `pars[0:n_gaussians]`: Amplitudes of the Gaussian peaks.
        - `pars[n_gaussians:2*n_gaussians]`: Centers of the Gaussian peaks.
        - `pars[2*n_gaussians:3*n_gaussians]`: Sigmas of the Gaussian peaks.
        - `pars[-2]`: Slope of the linear background.
        - `pars[-1]`: Intercept of the linear background.

    Returns
    -------
    numpy.ndarray
        Output of the combined Gaussian functions with the linear background.
    """
    n_gaussians = (len(pars) - 2) // 3
    amplitudes = pars[:n_gaussians]
    centers = pars[n_gaussians:2 * n_gaussians]
    sigmas = pars[2 * n_gaussians:3 * n_gaussians]
    slope = pars[-2]
    intercept = pars[-1]

    # Compute the linear background
    background = Linear(x, slope, intercept)

    # Compute the sum of Gaussian functions
    x_tiled = np.tile(x, (n_gaussians, 1)).T
    gaussians = Gaussian(x_tiled, amplitude=amplitudes, center=centers, sigma=sigmas)
    gaussians_sum = np.sum(gaussians, axis=1)

    return background + gaussians_sum

def CombGaussiansLS(pars, x, y):
    """
    Computes the residuals for least-squares fitting of a combination of Gaussian 
    functions with a linear background.

    Parameters
    ----------
    pars : list or numpy.ndarray
        Parameter vector for the Gaussian functions and linear background:
        - `pars[0:n_gaussians]`: Amplitudes of the Gaussian peaks.
        - `pars[n_gaussians:2*n_gaussians]`: Centers of the Gaussian peaks.
        - `pars[2*n_gaussians:3*n_gaussians]`: Sigmas of the Gaussian peaks.
        - `pars[-2]`: Slope of the linear background.
        - `pars[-1]`: Intercept of the linear background.
    x : numpy.ndarray
        Input values for the Gaussian functions.
    y : numpy.ndarray
        Observed data to fit.

    Returns
    -------
    numpy.ndarray
        Residuals between the observed data and the model.
    """    
    n_gaussians = (len(pars) - 2) // 3
    amplitudes = pars[:n_gaussians]
    centers = pars[n_gaussians:2 * n_gaussians]
    sigmas = pars[2 * n_gaussians:3 * n_gaussians]
    slope = pars[-2]
    intercept = pars[-1]

    # Compute the linear background
    background = Linear(x[:, 0], slope, intercept)

    # Compute the sum of Gaussian functions
    gaussians = Gaussian(x, amplitude=amplitudes, center=centers, sigma=sigmas)
    gaussians_sum = np.sum(gaussians, axis=1)

    # Compute residuals
    residuals = background + gaussians_sum - y

    return residuals

def CombGaussiansq(x, pars):
    """
    Computes a combination of multiple Gaussian functions with a quadratic background.

    Parameters
    ----------
    x : numpy.ndarray
        Input value(s) for the Gaussian functions.
    pars : list or numpy.ndarray
        Parameter vector for the Gaussian functions and quadratic background:
        - `pars[0:n_gaussians]`: Amplitudes of the Gaussian peaks.
        - `pars[n_gaussians:2*n_gaussians]`: Centers of the Gaussian peaks.
        - `pars[2*n_gaussians:3*n_gaussians]`: Sigmas of the Gaussian peaks.
        - `pars[-3]`: Coefficient for the quadratic term in the background.
        - `pars[-2]`: Coefficient for the linear term in the background.
        - `pars[-1]`: Constant term in the background.

    Returns
    -------
    numpy.ndarray
        Output of the combined Gaussian functions with the quadratic background.
    """
    n_gaussians = (len(pars) - 3) // 3
    amplitudes = pars[:n_gaussians]
    centers = pars[n_gaussians:2 * n_gaussians]
    sigmas = pars[2 * n_gaussians:3 * n_gaussians]
    a = pars[-3]
    b = pars[-2]
    c = pars[-1]

    # Compute the quadratic background
    background = Quadratic(x, a, b, c)

    # Compute the sum of Gaussian functions
    x_tiled = np.tile(x, (n_gaussians, 1)).T
    gaussians = Gaussian(x_tiled, amplitude=amplitudes, center=centers, sigma=sigmas)
    gaussians_sum = np.sum(gaussians, axis=1)

    return background + gaussians_sum

def CombGaussiansLSq(pars, x, y):
    """
    Computes the residuals for least-squares fitting of a combination of Gaussian 
    functions with a quadratic background.

    Parameters
    ----------
    pars : list or numpy.ndarray
        Parameter vector for the Gaussian functions and quadratic background:
        - `pars[0:n_gaussians]`: Amplitudes of the Gaussian peaks.
        - `pars[n_gaussians:2*n_gaussians]`: Centers of the Gaussian peaks.
        - `pars[2*n_gaussians:3*n_gaussians]`: Sigmas of the Gaussian peaks.
        - `pars[-3]`: Coefficient for the quadratic term in the background.
        - `pars[-2]`: Coefficient for the linear term in the background.
        - `pars[-1]`: Constant term in the background.
    x : numpy.ndarray
        Input values for the Gaussian functions.
    y : numpy.ndarray
        Observed data to fit.

    Returns
    -------
    numpy.ndarray
        Residuals between the observed data and the model.
    """
    n_gaussians = (len(pars) - 3) // 3
    amplitudes = pars[:n_gaussians]
    centers = pars[n_gaussians:2 * n_gaussians]
    sigmas = pars[2 * n_gaussians:3 * n_gaussians]
    a = pars[-3]
    b = pars[-2]
    c = pars[-1]

    # Compute the quadratic background
    background = Quadratic(x[:, 0], a, b, c)

    # Compute the sum of Gaussian functions
    gaussians = Gaussian(x, amplitude=amplitudes, center=centers, sigma=sigmas)
    gaussians_sum = np.sum(gaussians, axis=1)

    # Compute residuals
    residuals = background + gaussians_sum - y

    return residuals

def GetPeaksInfo(res):
    """
    Extracts peak information (areas, positions, and FWHMs) from a parameter vector.

    Parameters
    ----------
    res : numpy.ndarray or list
        A parameter vector containing the areas, positions, and FWHMs of peaks, 
        typically generated from peak fitting.

    Returns
    -------
    tuple
        A tuple of three arrays:
        - `Area`: Peak areas.
        - `Pos`: Peak positions.
        - `FWHM`: Peak full width at half maximum (FWHM).
    """
    n_peaks = (len(res) - 2) // 3
    Area = res[:n_peaks]
    Pos = res[n_peaks:2 * n_peaks]
    FWHM = res[2 * n_peaks:3 * n_peaks]
    return Area, Pos, FWHM


def Cagliotti(twotheta, U, V, W):
    """
    Computes the FWHM of a peak as a function of 2θ using the Cagliotti function.

    Parameters
    ----------
    twotheta : numpy.ndarray or float
        Two-theta values (in degrees).
    U : float
        Cagliotti parameter for the quadratic term.
    V : float
        Cagliotti parameter for the linear term.
    W : float
        Cagliotti parameter for the constant term.

    Returns
    -------
    numpy.ndarray or float
        FWHM values computed using the Cagliotti function.
    """
    return np.sqrt(U * tand(twotheta / 2.0)**2 + V * tand(twotheta / 2.0) + W)

def CagliottiLS(pars, tantth, y):
    """
    Computes residuals for least-squares fitting of the Cagliotti function.

    Parameters
    ----------
    pars : list or numpy.ndarray
        Parameters of the Cagliotti function:
        - `pars[0]`: U parameter.
        - `pars[1]`: V parameter.
        - `pars[2]`: W parameter.
    tantth : numpy.ndarray
        Tangent of half the two-theta values.
    y : numpy.ndarray
        Observed FWHM values.

    Returns
    -------
    numpy.ndarray
        Residuals between the observed FWHM values and the model.
    """
    return y - Cagliotti(tantth[:, 0], pars[0], pars[1], pars[2])


def savepeakfits(fn, Areas, Pos, Sigma):
    """
    Saves peak fitting results to an HDF5 file.

    Parameters
    ----------
    fn : str
        Filename for the HDF5 file (use .h5 or .hdf5 extension).
    Areas : dict
        Dictionary containing peak areas, with keys corresponding to peak indices.
    Pos : dict
        Dictionary containing peak positions, with keys corresponding to peak indices.
    Sigma : dict
        Dictionary containing peak FWHMs, with keys corresponding to peak indices.

    Notes
    -----
    - Each peak's area, position, and FWHM is saved as a separate dataset in the HDF5 file.
    - The function overwrites any existing file with the same name.
    """
    with h5py.File(fn, "w") as h5f:
        for ii in Areas.keys():
            h5f.create_dataset(f'Peak_Area_{ii}', data=Areas[ii])
            h5f.create_dataset(f'Peak_Position_{ii}', data=Pos[ii])
            h5f.create_dataset(f'Peak_FWHM_{ii}', data=Sigma[ii])

	
def loadpeakfits(fn, peaks):
    """
    Loads peak fitting results from an HDF5 file.

    This function reads peak areas, positions, and FWHMs, as well as background 
    parameters from an HDF5 file. The function assumes that the data is stored 
    in a format compatible with the `savepeakfits` function.

    Parameters
    ----------
    fn : str
        Filename of the HDF5 file (use .h5 or .hdf5 extension).
    peaks : int
        Number of peaks stored in the file.

    Returns
    -------
    tuple
        A tuple containing:
        - `Areas` (dict): Peak areas, where keys correspond to peak indices.
        - `Pos` (dict): Peak positions, where keys correspond to peak indices.
        - `Sigma` (dict): Peak FWHMs, where keys correspond to peak indices.
        - `Bkga` (numpy.ndarray): Background parameter `a`.
        - `Bkgb` (numpy.ndarray): Background parameter `b`.
        - `Bkgc` (numpy.ndarray or None): Background parameter `c` (if present, otherwise None).

    Notes
    -----
    - Background parameter `Bkgc` is optional. If not present in the file, its value 
      in the return tuple will be `None`.

    """
    Areas = {}
    Pos = {}
    Sigma = {}
    Bkga = None
    Bkgb = None
    Bkgc = None

    with h5py.File(fn, 'r') as f:
        # Load peak information
        for ii in range(peaks):
            Areas[ii] = f[f'/Peak_Area_{ii}'][:]
            Pos[ii] = f[f'/Peak_Position_{ii}'][:]
            Sigma[ii] = f[f'/Peak_FWHM_{ii}'][:]

        # Load background parameters
        Bkga = f['/Bkga'][:]
        Bkgb = f['/Bkgb'][:]

        # Load optional Bkgc parameter
        if '/Bkgc' in f:
            Bkgc = f['/Bkgc'][:]

    return Areas, Pos, Sigma, Bkga, Bkgb, Bkgc


def subtract_linear_background(data_roi, tth_roi):
    """
    Subtracts a linear background from the last axis of a 3D array.

    The background is estimated by defining a straight line connecting 
    the first and last intensity values of each spectrum along the last axis.

    Parameters
    ----------
    data_roi : numpy.ndarray
        Input 3D array containing the spectral data. The function assumes 
        the spectral channel (energy/tth) is along the last axis 
        (e.g., shape [dim1, dim2, channels]).
    tth_roi : numpy.ndarray
        1D array representing the x-axis values (e.g., 2-theta angles) 
        corresponding to the last dimension of `data_roi`.

    Returns
    -------
    numpy.ndarray
        The background-subtracted data, with the same shape as `data_roi`.
    """
    x1 = tth_roi[0]
    x2 = tth_roi[-1]
    
    y1 = data_roi[:, :, 0]
    y2 = data_roi[:, :, -1]
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Broadcast slope and intercept to match data_roi shape for subtraction
    background = slope[..., None] * tth_roi + intercept[..., None]
    
    return data_roi - background

def fit_gaussian_map(data, x_axis, threshold=1e-3, sigma_bounds=(0.01, 1.0)):
    """
    Fits a Gaussian function to the spectral dimension of a 3D dataset for every pixel.

    Iterates through the spatial dimensions (x, y) of the input data and fits a 
    Gaussian peak to the signal along the last axis. Pixels with maximum intensity 
    below the specified threshold are skipped.

    Parameters
    ----------
    data : numpy.ndarray
        Input 3D array (e.g., [x, y, channels]) containing the spectral data 
        to be fitted.
    x_axis : numpy.ndarray
        1D array representing the x-axis values (e.g., 2-theta or energy) 
        corresponding to the last dimension of `data`.
    threshold : float, optional, default=1e-3
        Intensity threshold. Pixels where the maximum signal is below this 
        value will be skipped (parameters remain 0).
    sigma_bounds : tuple, optional, default=(0.01, 1.0)
        A tuple of (min, max) defining the lower and upper bounds for the 
        sigma (width) parameter of the Gaussian.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (x, y, 3) containing the fitted parameters for 
        each pixel. The last axis corresponds to [Amplitude, Mean (mu), Sigma].
        Returns zeros for pixels where the fit failed or signal was below threshold.
    """
    nx, ny, _ = data.shape
    params = np.zeros((nx, ny, 3))  # A, mu, sigma

    # Fitting bounds: 
    # Amplitude: 0 to inf
    # Position (mu): constrained to the range of x_axis
    # Sigma: constrained by sigma_bounds
    lower_bounds = [0, x_axis[0], sigma_bounds[0]]
    upper_bounds = [np.inf, x_axis[-1], sigma_bounds[1]]
    bounds = (lower_bounds, upper_bounds)

    # Loop over all pixels
    for ix in tqdm(range(nx), desc="Fitting Gaussians"):
        for iy in range(ny):
            y = data[ix, iy, :]
            
            # Skip pixels with very low signal
            if y.max() < threshold:
                continue
            
            # Initial guess: [amplitude, position, width]
            guess = [y.max(), x_axis[np.argmax(y)], 0.01]
            
            try:
                popt, _ = curve_fit(Gaussian, x_axis, y, p0=guess, bounds=bounds)
                params[ix, iy, :] = popt
            except RuntimeError:
                pass  # Fit failed, parameters remain 0

    return params