# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony Vamvakeros
"""
#%%

import numpy as np
from skimage.transform import iradon, radon
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass
from tqdm import tqdm
from scipy.fft import rfft
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import algotom.util.utility as util
import algotom.prep.removal as remo

def radonvol(vol, scan = 180, theta=None):
    
    '''
    Computes the Radon transform of a stack of 2D images or a single 2D image.
    
    The Radon transform is calculated along specified projection angles for each 
    2D slice in the input volume. This function supports both 2D and 3D input 
    data, where the third dimension in the 3D case corresponds to either the 
    z-axis or spectral dimension.

    Parameters
    ----------
    vol : ndarray
        Input volume. Can be either:
        - A 2D array (H x W) representing a single image.
        - A 3D array (H x W x D) representing a stack of D images.
    scan : int, optional
        Total range of angles (in degrees) over which projections are computed.
        Default is 180 degrees.
    theta : array-like, optional
        Array of projection angles (in degrees). If not provided, 
        the angles are evenly spaced over the range defined by `scan`.

    Returns
    -------
    ndarray
        Sinogram or sinogram volume:
        - For a 2D input: A 2D array of shape (len(theta), W).
        - For a 3D input: A 3D array of shape (H, len(theta), D).

    Notes
    -----
    - The Radon transform is calculated using `skimage.transform.radon`.
    - The output dimensions depend on the input shape:
      - For 2D input: `(len(theta), width of the input image)`
      - For 3D input: `(height of the input image stack, len(theta), depth of the stack)`
    '''
    if theta is None:
        theta = np.arange(0, scan, scan/vol.shape[0])    
    nproj = len(theta)
    if len(vol.shape)>2:
        s = np.zeros((vol.shape[0], nproj, vol.shape[2]))    
        for ii in tqdm(range(s.shape[2])):
            s[:,:,ii] = radon(vol[:,:,ii], theta)
    elif len(vol.shape)==2:
        s = radon(vol, theta)
    print('The dimensions of the sinogram volume are ', s.shape)
    return(s)
        
def fbpvol(svol, scan = 180, theta=None, nt = None):
    
    """
    Reconstructs a stack of images or a single image from sinograms using the 
    filtered backprojection (FBP) algorithm.
    
    The function processes a 2D sinogram or a 3D stack of sinograms, where the 
    third dimension represents either the z-axis or spectral dimension. 
    Reconstruction is performed for each slice using the filtered backprojection 
    method from `skimage.transform.iradon`.

    Parameters
    ----------
    svol : ndarray
        Input sinogram or sinogram volume. Can be either:
        - A 2D array (n_projections x detector_width) representing a single sinogram.
        - A 3D array (n_rows x n_projections x n_depths) representing a stack of sinograms.
    scan : int, optional
        Total range of projection angles (in degrees) over which the sinograms 
        were acquired. Default is 180 degrees.
    theta : array-like, optional
        Array of projection angles (in degrees). If not provided, 
        the angles are evenly spaced over the range defined by `scan`.
    nt : int, optional
        Desired size of the reconstructed images (number of pixels along one 
        dimension). If not provided, it defaults to the number of rows in `svol`.

    Returns
    -------
    ndarray
        Reconstructed image or volume:
        - For a 2D input: A 2D array of shape (nt, nt).
        - For a 3D input: A 3D array of shape (nt, nt, n_depths).

    Notes
    -----
    - The reconstruction is performed using `skimage.transform.iradon`.
    - The `circle` parameter in `iradon` is set to `True`, assuming the object 
      fits entirely within the reconstruction area.
    - The output dimensions depend on the input:
      - For 2D sinograms: `(nt, nt)`.
      - For 3D sinogram stacks: `(nt, nt, n_depths)`.

    """
    if nt is None:
        nt = svol.shape[0]
    nproj = svol.shape[1]
    
    if theta is None:
        theta = np.arange(0, scan, scan/nproj)
    
    if len(svol.shape)>2:
    
        vol = np.zeros((nt, nt, svol.shape[2]))
        
        for ii in tqdm(range(svol.shape[2])):
            
            vol[:,:,ii] = iradon(svol[:,:,ii], theta, nt, circle = True)
                
    elif len(svol.shape)==2:
        
        vol = iradon(svol, theta, nt, circle = True)
    
    print('The dimensions of the reconstructed volume are ', vol.shape)
        
    return(vol)

    
def create_ramp_filter(s, ang):
    """
    Creates a ramp filter for the sinograms based on the detector width and angles.

    This function computes a 2D ramp filter, where the frequency components are 
    scaled by their absolute values. The filter is designed for the input sinograms 
    and repeated for each angle in the projection.

    Parameters
    ----------
    s : numpy.ndarray
        Input sinograms with shape (detector elements, projections).
    ang : numpy.ndarray or list
        Array or list of projection angles.

    Returns
    -------
    numpy.ndarray
        A 2D ramp filter with shape (len(ang), detector elements).

    Notes
    -----
    - The ramp filter is calculated using a frequency-domain approach.
    - The filter is repeated along the angle dimension to match the input sinograms.

    """
    N1 = s.shape[1]  # Number of detector elements
    freqs = np.linspace(-1, 1, N1)  # Normalized frequency range
    myFilter = np.abs(freqs)  # Ramp filter in frequency domain
    myFilter = np.tile(myFilter, (len(ang), 1))  # Repeat filter for all angles
    return myFilter

def ramp(detector_width):
    """
    Computes a 1D ramp filter in the frequency domain.

    This function generates a ramp filter for a given detector width using 
    frequency components. The filter is symmetric and designed for use in 
    reconstruction algorithms such as filtered backprojection.

    Parameters
    ----------
    detector_width : int
        Number of detector elements.

    Returns
    -------
    numpy.ndarray
        A 1D ramp filter with length equal to the detector width.

    Notes
    -----
    - The filter is constructed by linearly scaling frequencies up to the Nyquist 
      frequency and symmetrically decreasing beyond it.
    - The output is a float32 array for numerical efficiency.

    """    
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width / 2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0] / 2.0)) * frequency_spacing)
    return filter_array.astype(np.float32)


def paralleltomo(N, theta, p, w):
    """
    Creates a 2D tomography test problem using parallel beams.

    This function generates a sparse matrix `A` representing the forward projection 
    operator for a 2D domain discretized into N x N cells. The problem is defined 
    using `p` parallel rays for each angle in `theta`.

    Parameters
    ----------
    N : int
        Number of discretization intervals in each dimension, resulting in an N x N domain.
    theta : array-like
        Array of angles in degrees.
    p : int, optional
        Number of parallel rays for each angle. Default is `round(sqrt(2) * N)`.
    w : float, optional
        Width of the ray fan (distance from the first ray to the last). Default is `sqrt(2) * N`.

    Returns
    -------
    csr_matrix
        Sparse matrix `A` with shape (nA * p, N^2), where `nA` is the number of angles in `theta`.
        The rows represent rays, and the columns correspond to cells in the domain.

    Notes
    -----
    - The generated matrix `A` can be used in test problems for tomographic reconstruction.
    - This implementation assumes the rays pass through a square domain centered at the origin.

    References
    ----------
    - A. C. Kak and M. Slaney, "Principles of Computerized Tomographic Imaging," SIAM, Philadelphia, 2001.
    - Original MATLAB code: AIR Tools, DTU Informatics, June 21, 2011.

    """
    if p is None:
        p = round(np.sqrt(2) * N)
    if w is None:
        w = np.sqrt(2) * N

    nA = len(theta)  # Number of angles
    x0 = np.linspace(-w / 2, w / 2, p).reshape(-1, 1)
    y0 = np.zeros((p, 1))
    x = np.arange(-N / 2, N / 2 + 1)
    y = x

    rows, cols, vals = [], [], []
    thetar = np.deg2rad(theta)  # Convert angles to radians

    for i in range(nA):
        x0theta = np.cos(thetar[i]) * x0 - np.sin(thetar[i]) * y0
        y0theta = np.sin(thetar[i]) * x0 + np.cos(thetar[i]) * y0

        a = -np.sin(thetar[i])
        b = np.cos(thetar[i])

        for j in range(p):
            tx = (x - x0theta[j]) / a
            yx = b * tx + y0theta[j]
            ty = (y - y0theta[j]) / b
            xy = a * ty + x0theta[j]

            t = np.concatenate((tx, ty))
            xxy = np.concatenate((x, xy))
            yxy = np.concatenate((yx, y))

            I = np.argsort(t)
            t, xxy, yxy = t[I], xxy[I], yxy[I]

            I = (xxy >= -N / 2) & (xxy <= N / 2) & (yxy >= -N / 2) & (yxy <= N / 2)
            xxy, yxy = xxy[I], yxy[I]

            I = (np.abs(np.diff(xxy)) > 1e-10) | (np.abs(np.diff(yxy)) > 1e-10)
            xxy, yxy = xxy[:-1][I], yxy[:-1][I]

            d = np.sqrt(np.diff(xxy)**2 + np.diff(yxy)**2)
            if d.size > 0:
                xm = 0.5 * (xxy[:-1] + xxy[1:]) + N / 2
                ym = 0.5 * (yxy[:-1] + yxy[1:]) + N / 2

                col = np.floor(xm) * N + (N - np.floor(ym)) - 1
                row = i * p + j

                rows.extend([row] * len(d))
                cols.extend(col.astype(int))
                vals.extend(d)

    A = csr_matrix((vals, (rows, cols)), shape=(nA * p, N**2), dtype=np.float32)
    return A

        
def cgls(A, b, K = 25, plot=False):
    """
    Conjugate Gradient Least Squares (CGLS) method for solving linear systems.

    This function solves the least squares problem `Ax ≈ b` using the iterative 
    Conjugate Gradient Least Squares (CGLS) method. It also includes an option 
    to plot intermediate results during the iterations.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Sparse matrix representing the system matrix.
    b : numpy.ndarray
        Right-hand side vector of the system, with shape (m, 1).
    K : int, optional, default=25
        Maximum number of iterations for the CGLS method.
    plot : bool, optional, default=False
        If True, plots the reconstructed solution after each iteration.

    Returns
    -------
    numpy.ndarray
        Reconstructed solution `x` as a 2D array with shape `(sqrt(n), sqrt(n))`, 
        where `n` is the number of columns in `A`.

    Notes
    -----
    - The function assumes that `A` represents a square domain, and the solution 
      is reshaped into a square image.
    - Negative values in the solution are clipped to zero, and the result is 
      normalized to have a maximum value of 1.

    """
    # Ensure K is an integer
    k = int(K)

    # Number of pixels in the solution
    n = A.shape[1]
    npix = int(np.sqrt(n))

    # Initialization
    x = np.zeros((n, 1))  # Initial guess
    r = b.copy()
    d = csr_matrix.dot(A.T, r)  # Initial residual
    normr2 = np.dot(d.T, d)

    if plot:
        plt.figure()
        plt.clf()

    # CGLS iterations
    for j in range(k):
        # Update x and r vectors
        Ad = csr_matrix.dot(A, d)
        alpha = normr2 / (np.dot(Ad.T, Ad) + 1e-10)  # Avoid division by zero
        x += d * alpha
        r -= Ad * alpha

        # Update s and d vectors
        s = csr_matrix.dot(A.T, r)
        normr2_new = np.dot(s.T, s)
        beta = normr2_new / (normr2 + 1e-10)  # Avoid division by zero
        normr2 = normr2_new
        d = s + d * beta

        # Reshape and normalize for visualization
        xn = x.reshape((npix, npix))
        xn = np.clip(xn, 0, None)  # Set negative values to 0
        xn /= np.max(xn) if np.max(xn) > 0 else 1  # Normalize

        # Plot intermediate results if requested
        if plot:
            plt.imshow(xn, cmap='jet')
            plt.title(f"Iteration {j+1}")
            plt.colorbar()
            plt.pause(0.5)

    return xn

