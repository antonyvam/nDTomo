# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony Vamvakeros
"""
#%%

import numpy as np
from skimage.transform import iradon, radon
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import rotate
from scipy.sparse import csr_matrix, diags

    
def filter_sinogram(sinogram, filter_type='Ramp'):
    """
    Applies a frequency domain filter to a sinogram.

    Parameters
    ----------
    sinogram : ndarray
        2D array of shape (num_detectors, num_angles).
    filter_type : str
        Type of filter to apply. Options:
        - 'Ramp' : standard Ram-Lak filter
        - 'Shepp-Logan' : Ramp * sinc

    Returns
    -------
    filtered_sinogram : ndarray
        The filtered sinogram with the same shape as input.
    """
    n, n_theta = sinogram.shape
    f = fftfreq(n).reshape(-1, 1)

    # Ramp filter
    ramp = np.abs(f)

    if filter_type == 'Ramp':
        filt = ramp
    elif filter_type == 'Shepp-Logan':
        filt = ramp * np.sinc(f / (2 * f.max()))
    else:
        raise ValueError("Unsupported filter_type. Choose 'Ramp' or 'Shepp-Logan'.")

    # FFT along detector axis (rows)
    sino_fft = fft(sinogram, axis=0)
    sino_fft_filtered = sino_fft * filt
    filtered_sinogram = np.real(ifft(sino_fft_filtered, axis=0))

    return filtered_sinogram

def forwardproject(image, angles, num_detectors=None):
    """
    Simulates a forward projection (Radon transform) of a 2D image.

    Parameters
    ----------
    image : ndarray
        2D input image (square or rectangular).
    angles : ndarray
        1D array of projection angles in degrees.
    num_detectors : int, optional
        Number of detectors in the output sinogram. If None, uses image height.

    Returns
    -------
    sinogram : ndarray
        2D sinogram of shape (num_detectors, num_angles).
    """
    image = np.asarray(image)
    img_size = image.shape[0]
    if num_detectors is None:
        num_detectors = img_size

    sinogram = np.zeros((num_detectors, len(angles)), dtype=np.float32)

    for i, theta in enumerate(angles):
        # Rotate the image to simulate the projection direction
        rotated = rotate(image, angle=-theta, reshape=False, order=3)

        # Sum along columns to simulate detector response
        projection = np.sum(rotated, axis=0)

        # Center crop or pad to match desired number of detectors
        if len(projection) == num_detectors:
            sinogram[:, i] = projection
        elif len(projection) > num_detectors:
            offset = (len(projection) - num_detectors) // 2
            sinogram[:, i] = projection[offset:offset + num_detectors]
        else:
            pad = (num_detectors - len(projection)) // 2
            sinogram[:, i] = np.pad(projection, (pad, num_detectors - len(projection) - pad), mode='constant')

    return sinogram

def backproject(sinogram, angles, output_size=None, normalize=True):
    """
    Performs filtered or unfiltered backprojection of a sinogram.

    Parameters
    ----------
    sinogram : ndarray
        2D array of shape (num_detectors, num_angles).
    angles : ndarray
        1D array of projection angles in degrees.
    output_size : int, optional
        Size (width and height) of the reconstructed image.
        If None, use num_detectors from sinogram.

    Returns
    -------
    reconstruction : ndarray
        2D reconstructed image.
    """
    n_detectors, n_angles = sinogram.shape
    if output_size is None:
        output_size = n_detectors

    reconstruction = np.zeros((output_size, output_size), dtype=np.float32)

    for i, theta in enumerate(angles):
        projection = sinogram[:, i]

        # Expand the 1D projection to a 2D image (replicate across columns)
        projection_2d = np.tile(projection[:, np.newaxis], (1, output_size))

        # Rotate the 2D projection to the current angle
        rotated = rotate(projection_2d, angle= theta + 90, reshape=False, order=3)

        # Accumulate into reconstruction
        reconstruction += rotated

    # Normalize#
    if normalize:
        reconstruction *= np.pi / (n_angles)

    return reconstruction


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

        
def sirt(A, b, x0=None, n_iter=20, relax=1.0):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT) with row and column normalization.

    This function implements a general version of SIRT using matrix-based normalization
    inspired by Cimmino/CAV-style scaling. It solves the linear system A @ x ≈ b iteratively.

    Each iteration computes the global residual and applies a normalized correction:
        x ← x + relax * D * A.T * M * (b - A @ x)

    Parameters
    ----------
    A : scipy.sparse matrix (CSR or CSC), shape (m, n)
        The system matrix, typically sparse and non-negative. Rows represent projection rays;
        columns represent image pixels.
    b : ndarray, shape (m,)
        The measured projection data (sinogram flattened to 1D).
    x0 : ndarray or None, shape (n,), optional
        Initial guess for the solution. Defaults to zeros if None.
    n_iter : int, optional
        Number of SIRT iterations to perform. More iterations yield smoother convergence.
    relax : float, optional
        Relaxation parameter controlling update magnitude. Values between 0.5–1.0 are typical.

    Returns
    -------
    x : ndarray, shape (n,)
        The reconstructed solution vector (flattened image).

    Notes
    -----
    - The row normalization matrix M scales residual contributions by the inverse of row energy:
        M[i, i] = 1 / ||A[i, :]||²
    - The column normalization matrix D scales the updates to balance pixel sensitivity:
        D[j, j] = 1 / ||A[:, j]||²
    - This global scheme is slower than CGLS but more stable for noisy or incomplete data.
    - Particularly useful in tomographic reconstruction where balancing ray coverage is critical.
    """
    A = A.tocsr()
    m, n = A.shape

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Row normalization: M = diag(1 / ||a_i||^2)
    row_norms = np.array(A.power(2).sum(axis=1)).flatten()
    row_norms[row_norms == 0] = 1
    M_inv = 1.0 / row_norms  # shape (m,)

    # Column normalization: D = diag(1 / ||a_j||^2)
    col_norms = np.array(A.power(2).sum(axis=0)).flatten()
    col_norms[col_norms == 0] = 1
    D_inv = 1.0 / col_norms  # shape (n,)

    D = diags(D_inv)
    M = diags(M_inv)

    for _ in range(n_iter):
        r = b - A @ x                      # residual
        x += relax * D @ (A.T @ (M @ r))   # apply normalized update

    return x


def cgls(A, b, x0=None, n_iter=10):
    """
    Conjugate Gradient Least Squares (CGLS) algorithm for solving Ax = b.

    This method minimizes the least-squares objective:
        ||Ax - b||^2

    It is equivalent to applying the Conjugate Gradient method to the 
    normal equations:
        AᵀA x = Aᵀb

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (m, n)
        The system matrix. Can be dense or sparse.
    b : ndarray, shape (m,)
        The right-hand side vector (e.g., sinogram or projection data).
    x0 : ndarray or None, shape (n,), optional
        Initial guess for the solution. If None, defaults to zeros.
    n_iter : int, optional
        Number of iterations to perform. Default is 10.

    Returns
    -------
    x : ndarray, shape (n,)
        The reconstructed solution vector.

    Notes
    -----
    - CGLS converges faster than basic iterative methods like ART or SIRT,
      especially for well-conditioned problems.
    - It avoids explicitly forming AᵀA, which can be ill-conditioned or
      expensive to compute.
    - This implementation does not support stopping criteria based on tolerance.
    """
    
    if x0 is None:
        x = np.zeros(A.shape[1])
    else:
        x = x0.copy()

    r = b - A @ x
    p = A.T @ r
    d = p.copy()
    delta_new = np.dot(d, d)

    for _ in range(n_iter):
        q = A @ d
        alpha = delta_new / np.dot(q, q)
        x += alpha * d
        r -= alpha * q
        s = A.T @ r
        delta_old = delta_new
        delta_new = np.dot(s, s)
        beta = delta_new / delta_old
        d = s + beta * d

    return x



def cgls_functional(sinogram, angles, x0=None, n_iter=10):
    """
    CGLS solver using forwardproject and backproject, matching the vectorized logic.

    Parameters
    ----------
    sinogram : ndarray
        Input sinogram of shape (num_detectors, num_angles).
    angles : ndarray
        1D array of projection angles in degrees.
    x0 : ndarray or None
        Initial image guess. If None, initialized to zeros.
    n_iter : int
        Number of CGLS iterations.

    Returns
    -------
    image : ndarray
        Reconstructed 2D image of shape (num_detectors, num_detectors).
    """
    N, n_angles = sinogram.shape

    def forward(x):  # returns sinogram shape
        return forwardproject(x.reshape((N, N)), angles).ravel()

    def backward(y):  # y is flattened sinogram
        y2d = y.reshape((N, n_angles))
        return backproject(y2d, angles, output_size=N, normalize=False).ravel()

    b = sinogram.ravel()

    if x0 is None:
        x = np.zeros(N * N, dtype=np.float32)
    else:
        x = x0.ravel().copy()

    r = b - forward(x)
    p = backward(r)
    d = p.copy()
    delta_new = np.dot(d, d)

    for _ in range(n_iter):
        q = forward(d)
        alpha = delta_new / np.dot(q, q)
        x += alpha * d
        r -= alpha * q
        s = backward(r)
        delta_old = delta_new
        delta_new = np.dot(s, s)
        beta = delta_new / delta_old
        d = s + beta * d

    return x.reshape((N, N))


def sirt_functional(sinogram, angles, x0=None, n_iter=20, relax=1.0, epsilon=1e-6):
    """
    SIRT solver using forwardproject and backproject, with image size inferred from sinogram.

    Parameters
    ----------
    sinogram : ndarray
        Input sinogram of shape (num_detectors, num_angles).
    angles : ndarray
        1D array of projection angles in degrees.
    x0 : ndarray or None
        Initial guess for the image. If None, initialized to zeros.
    n_iter : int
        Number of SIRT iterations.
    relax : float
        Relaxation factor.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    image : ndarray
        Reconstructed 2D image.
    """
    N = sinogram.shape[0]

    def forward(x): return forwardproject(x.reshape((N, N)), angles)
    def backward(y): return backproject(y, angles, output_size=N, normalize=False)

    b = sinogram
    W_voxel = backward(forward(np.ones((N, N), dtype=np.float32)))

    if x0 is None:
        x = np.zeros((N, N), dtype=np.float32)
    else:
        x = x0.copy()

    for _ in range(n_iter):
        residual = b - forward(x)
        correction = backward(residual) / (W_voxel + epsilon)
        x += relax * correction

    return x
