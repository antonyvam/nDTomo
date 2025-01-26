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

def airrem(sinograms, ofs=1, pbar=True, method="both", coli=0, colf=None):
    """
    Removes background signal (air signal) from sinograms.
    
    This method subtracts the background signal from each projection in the sinograms.
    The background signal can be estimated from the top row, bottom row, both rows, 
    or specific columns in these rows. It supports both 2D and 3D sinograms, where 
    the third dimension represents the z-axis or spectral dimension.

    Parameters
    ----------
    sinograms : ndarray
        Input sinograms. Can be either:
        - A 2D array (translation_steps x projections) for a single sinogram.
        - A 3D array (translation_steps x projections x z) for a stack of sinograms.
    ofs : int, optional
        Offset defining the number of rows at the top and/or bottom of the sinogram 
        used to calculate the background signal. Default is 1.
    pbar : bool, optional
        If True, displays a progress bar during the background removal process. 
        Default is True.
    method : {"both", "top", "bottom"}, optional
        Method for selecting the rows used to calculate the background signal:
        - "both": Uses the top `ofs` rows and the bottom `ofs` rows (default).
        - "top": Uses only the top `ofs` rows.
        - "bottom": Uses only the bottom `ofs` rows.
    coli : int, optional
        Starting column index for selecting a subset of columns in the top or 
        bottom rows. Default is 0 (start from the first column).
    colf : int, optional
        Ending column index (exclusive) for selecting a subset of columns in the 
        top or bottom rows. Default is None (use all columns).

    Returns
    -------
    ndarray
        Sinograms with the background signal removed. The output has the same shape 
        as the input.

    Notes
    -----
    - Background signal is calculated based on the selected rows and columns as 
      defined by the `method`, `coli`, and `colf` parameters.
    - If `sinograms` is 3D, the background removal is performed for each slice 
      along the z-axis (or spectral dimension).
    """
    di = sinograms.shape
    loop_range = tqdm(range(sinograms.shape[1])) if pbar else range(sinograms.shape[1])

    if colf is None:
        colf = sinograms.shape[1]  # Use all columns if colf is not specified
    
    if len(di) > 2:  # 3D sinograms
        for ii in loop_range:
            if method == "both":
                air = (
                    np.mean(sinograms[0:ofs, ii, coli:colf], axis=(0, 1)) +
                    np.mean(sinograms[-ofs:, ii, coli:colf], axis=(0, 1))
                ) / 2
            elif method == "top":
                air = np.mean(sinograms[0:ofs, ii, coli:colf], axis=(0, 1))
            elif method == "bottom":
                air = np.mean(sinograms[-ofs:, ii, coli:colf], axis=(0, 1))
            sinograms[:, ii, :] -= air

    elif len(di) == 2:  # 2D sinograms
        for ii in range(sinograms.shape[1]):
            if method == "both":
                air = (
                    np.mean(sinograms[0:ofs, coli:colf], axis=0) +
                    np.mean(sinograms[-ofs:, coli:colf], axis=0)
                ) / 2
            elif method == "top":
                air = np.mean(sinograms[0:ofs, coli:colf], axis=0)
            elif method == "bottom":
                air = np.mean(sinograms[-ofs:, coli:colf], axis=0)
            sinograms[:, ii] -= air

    return sinograms

def scalesinos(sinograms, pbar=False):
    
    """
    Normalizes a sinogram or sinogram volume based on total intensity per projection.
    
    This method assumes that the total intensity per projection is constant and 
    scales each projection to normalize the intensity. It supports both 2D and 3D 
    sinograms, where the third dimension in the 3D case represents the z-axis or 
    spectral dimension.

    Parameters
    ----------
    sinograms : ndarray
        Input sinograms. Can be either:
        - A 2D array (translations x projections) for a single sinogram.
        - A 3D array (translations x projections x z) for a stack of sinograms.
    pbar : bool, optional
        If True, displays a progress bar during the normalization process. 
        Default is False.

    Returns
    -------
    ndarray
        Normalized sinograms. The output has the same shape as the input.

    Notes
    -----
    - For 3D sinograms, normalization is performed slice-by-slice along the 
      z-axis (or spectral dimension).
    - The normalization factor for each projection is calculated as the ratio of 
      the total intensity per projection to the maximum total intensity across 
      all projections.
    """
    
    di = sinograms.shape
    loop_range = tqdm(range(sinograms.shape[1])) if pbar else range(sinograms.shape[1])

    if len(di) > 2:  # 3D sinograms
        # Summed scattering intensity per linescan
        ss = np.sum(sinograms, axis=2)  # Sum along the z-axis
        scint = np.sum(ss, axis=0)  # Total intensity per projection
        # Scale factors
        sf = scint / np.max(scint)
        
        # Normalize the sinogram data
        for jj in loop_range:
            sinograms[:, jj, :] /= sf[jj]

    elif len(di) == 2:  # 2D sinograms
        # Summed scattering intensity per linescan
        scint = np.sum(sinograms, axis=0)  # Total intensity per projection
        # Scale factors
        sf = scint / np.max(scint)
        
        # Normalize the sinogram data
        for jj in loop_range:
            sinograms[:, jj] /= sf[jj]

    return sinograms


def sinocomcor(sinograms, interp=True):
    """
    Corrects sinograms for motor jitter by aligning them based on their center of mass.
    
    This method adjusts the sinograms to correct for translation misalignments 
    (motor jitter) by interpolating each projection to align its center of mass. 
    It supports both 2D and 3D sinograms, where the third dimension in the 3D case 
    represents the z-axis or spectral dimension.

    Parameters
    ----------
    sinograms : ndarray
        Input sinograms. Can be either:
        - A 2D array (translations x projections) for a single sinogram.
        - A 3D array (translations x projections x z) for a stack of sinograms.
    interp : bool, optional
        If True, linear interpolation is performed with no extrapolation.
        If False, extrapolated values outside the original range are set to 0.
        Default is True.

    Returns
    -------
    ndarray
        Sinograms corrected for motor jitter. The output has the same shape as the input.

    Notes
    -----
    - The correction is based on the center of mass (COM) of each projection, calculated 
      using `scipy.ndimage.center_of_mass`.
    - The first projection is used as the reference for alignment.
    - Linear interpolation (`np.interp`) is used to shift each projection based on 
      its COM. Extrapolation behavior depends on the `interp` parameter.

    """
    di = sinograms.shape

    # Sum along the spectral axis if sinograms are 3D
    if len(di) > 2:
        ss = np.sum(sinograms, axis=2)
    else:
        ss = np.copy(sinograms)

    # Calculate center of mass for each projection
    com = np.zeros((ss.shape[1], 1))
    for ii in range(ss.shape[1]):
        com[ii, :] = center_of_mass(ss[:, ii])

    # Adjust COM relative to the first projection
    com = com - com[0]

    # Create an empty array for corrected sinograms
    sn = np.zeros_like(sinograms)
    xold = np.arange(sn.shape[0])

    if len(di) == 2:  # For 2D sinograms
        for ii in tqdm(range(sn.shape[1]), desc="Correcting 2D sinograms"):
            xnew = xold + com[ii, :]
            if interp:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii])
            else:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii], left=0, right=0)

    elif len(di) > 2:  # For 3D sinograms
        for ll in tqdm(range(sinograms.shape[2]), desc="Correcting 3D sinograms"):
            for ii in range(sinograms.shape[1]):
                xnew = xold + com[ii, :]
                if interp:
                    sn[:, ii, ll] = np.interp(xnew, xold, sinograms[:, ii, ll])
                else:
                    sn[:, ii, ll] = np.interp(xnew, xold, sinograms[:, ii, ll], left=0, right=0)

    return sn

def sinocentering(sinograms, crsr=5, interp=True, scan=180, channels = None, pbar=True):
    """
    Centers sinograms by calculating and applying the center of rotation (COR) correction.

    This function identifies the center of rotation by comparing projections at 0° and 180°. 
    It adjusts the sinograms accordingly to align the projections. It supports both 2D and 
    3D sinograms, where 3D sinograms include spectral (Z) channels.

    Parameters
    ----------
    sinograms : numpy.ndarray
        Input sinogram(s). 
        - 2D array: A single sinogram with shape (detector elements, projections).
        - 3D array: A stack of sinograms with shape (detector elements, projections, Z/spectral channels).

    crsr : float, optional, default=5
        Range for searching the center of rotation, expressed in detector element units.
        The search is conducted in the range [s.shape[0]/2 - crsr, s.shape[0]/2 + crsr].

    interp : bool, optional, default=True
        Specifies whether to use linear interpolation to adjust the sinogram for COR correction:
        - True: Use linear interpolation for smoother adjustments.
        - False: Use linear interpolation with edge extrapolation set to 0.

    scan : int, optional, default=180
        The scanning range of the projection angles:
        - 180: Use only the first half of the sinogram for COR determination.
        - 360: Use the entire sinogram for COR determination.

    channels : list or None, optional, default=None
        Specifies the spectral (Z) channels to consider when summing along the Z-axis:
        - None: Sum across all spectral channels.
        - List of indices: Use only the specified channels for the summation.

    pbar : bool, optional, default=True
        If True, display progress bars during the COR calculation and correction steps.
        If False, suppress progress bar output.

    Returns
    -------
    numpy.ndarray
        Corrected sinogram(s) after applying the calculated COR correction. 
        The shape of the returned array matches the input array, except for adjustments 
        in the detector elements due to the calculated COR shift.

    Notes
    -----
    - The center of rotation (COR) is determined by minimizing the standard deviation 
      of differences between the flipped projection at 180° and the projection at 0°.
    - For 3D sinograms, the COR correction is applied to each spectral channel individually.
    - The function uses `numpy.interp` for linear interpolation.

    """
    di = sinograms.shape
    if len(di) > 2:
        if channels is None:
            s = np.sum(sinograms, axis=2)
        else:
            s = np.sum(sinograms[:, :, channels[0]:channels[-1]], axis=2)
    else:
        s = np.copy(sinograms)

    # Adjust for scan range
    if scan == 360:
        s = s[:, :int(np.round(s.shape[1] / 2))]

    # Center of rotation search range
    cr = np.arange(s.shape[0] / 2 - crsr, s.shape[0] / 2 + crsr, 0.1)
    xold = np.arange(0, s.shape[0])
    st = []  # Standard deviations for COR candidates

    if pbar:
        print("Calculating the COR")
    loop_range = tqdm(range(len(cr))) if pbar else range(len(cr))
    for kk in loop_range:
        xnew = cr[kk] + np.arange(-np.ceil(s.shape[0] / 2), np.ceil(s.shape[0] / 2) - 1)
        sn = np.zeros((len(xnew), s.shape[1]), dtype="float32")
        for ii in range(s.shape[1]):
            if interp:
                sn[:, ii] = np.interp(xnew, xold, s[:, ii])
            else:
                sn[:, ii] = np.interp(xnew, xold, s[:, ii], left=0, right=0)
        re = sn[::-1, -1]
        st.append(np.std(sn[:, 0] - re))

    # Find the optimal center of rotation
    m = np.argmin(st)
    if pbar:
        print(f"Calculated COR: {cr[m]}")

    # Apply COR correction
    xnew = cr[m] + np.arange(-np.ceil(s.shape[0] / 2), np.ceil(s.shape[0] / 2) - 1)

    if pbar:
        print("Applying the COR correction")
    if len(di) > 2:  # 3D sinograms
        loop_range = tqdm(range(sinograms.shape[2])) if pbar else range(sinograms.shape[2])
        sn = np.zeros((len(xnew), sinograms.shape[1], sinograms.shape[2]), dtype="float32")
        for ll in loop_range:
            for ii in range(sinograms.shape[1]):
                if interp:
                    sn[:, ii, ll] = np.interp(xnew, xold, sinograms[:, ii, ll])
                else:
                    sn[:, ii, ll] = np.interp(xnew, xold, sinograms[:, ii, ll], left=0, right=0)
    else:  # 2D sinograms
        sn = np.zeros((len(xnew), sinograms.shape[1]), dtype="float32")
        for ii in range(sinograms.shape[1]):
            if interp:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii])
            else:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii], left=0, right=0)

    return sn

def zigzag_rows(s):
    """
    Rearranges the rows of a 3D array in a zigzag pattern.

    For all even-indexed rows (0-based) in the first dimension, the order of 
    the second dimension is reversed. This results in a zigzag pattern along 
    the first dimension.

    Parameters
    ----------
    s : numpy.ndarray
        Input 3D array with shape (rows, columns, depth).

    Returns
    -------
    numpy.ndarray
        Modified array with the rows rearranged in a zigzag pattern.

    Notes
    -----
    - The operation is performed in place; the input array is modified directly.
    - Only applies to 3D arrays. Ensure the input matches the expected shape.
    """
    s[0::2, :, :] = s[0::2, ::-1, :]
    return s

     
def zigzag_cols(s):
    """
    Rearranges the columns of a 3D array in a zigzag pattern along the first dimension.

    For all even-indexed columns (0-based) in the second dimension, the order of 
    the first dimension is reversed. This creates a zigzag pattern along the columns.

    Parameters
    ----------
    s : numpy.ndarray
        Input 3D array with shape (rows, columns, depth).

    Returns
    -------
    numpy.ndarray
        Modified array with the columns rearranged in a zigzag pattern.

    Notes
    -----
    - The operation is performed in place; the input array is modified directly.
    - Only applies to 3D arrays. Ensure the input matches the expected shape.

    Examples
    --------
    - If `s` has shape (rows, columns, depth):
        - For even-indexed columns, the rows are flipped.
    """
    s[:, 0::2, :] = s[::-1, 0::2, :]
    return s
    
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



def rot_center(thetasum):
    """
    Calculates the center of rotation (COR) of a sinogram using Fourier analysis.

    This method is based on the phase symmetry approach and requires the object to 
    be fully within the field of view at all projection angles. It also assumes 
    that the rotation axis is aligned with the axis of the 2D detector used to 
    acquire projection images.

    **Original Code Source**:
    https://github.com/everettvacek/PhaseSymmetry

    **Citation**:
    J. Synchrotron Rad. (2022). 29, https://doi.org/10.1107/S160057752101277

    Parameters
    ----------
    thetasum : numpy.ndarray
        A 2D array representing the thetasum (z, theta), typically obtained 
        by summing along one dimension of the sinogram.

    Returns
    -------
    float
        The calculated center of rotation (COR).

    Notes
    -----
    - The method assumes that the object is fully visible in the field of view 
      at all projection angles.
    - Assumes the rotation axis is perfectly aligned with the detector axis.
    - The method uses the real and imaginary components of the AC spatial 
      frequency to calculate the phase shift and determine the COR.
    """
    # Perform Fourier transform on the flattened thetasum array
    T = rfft(thetasum.ravel())

    # Extract real and imaginary components for the AC spatial frequency
    imag = T[thetasum.shape[0]].imag
    real = T[thetasum.shape[0]].real

    # Calculate phase and determine the center of rotation
    phase = np.arctan2(imag * np.sign(real), real * np.sign(real))
    COR = thetasum.shape[-1] / 2 - phase * thetasum.shape[-1] / (2 * np.pi)

    return COR

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


def myphantom(N):
    """
    Creates the modified Shepp-Logan phantom.

    This function generates the modified Shepp-Logan phantom with a discretization
    of N x N and returns it as a flattened vector. The intensities are adjusted 
    to yield higher contrast compared to the original Shepp-Logan phantom.

    Parameters
    ----------
    N : int
        Number of discretization intervals in each dimension. The phantom is 
        represented by an N x N grid.

    Returns
    -------
    numpy.ndarray
        A flattened vector representing the modified Shepp-Logan phantom with N^2 elements.

    Notes
    -----
    - The Shepp-Logan phantom consists of ellipses with specific intensities, 
      positions, and orientations.
    - The phantom is commonly used in tomographic reconstruction studies.

    References
    ----------
    - Peter Toft, "The Radon Transform - Theory and Implementation", PhD thesis, 
      DTU Informatics, Technical University of Denmark, June 1996.

    Original MATLAB code from AIR Tools, adapted to Python by Antony Vamvakeros.
    """
    # Define the ellipses: [A, a, b, x0, y0, phi]
    e = np.array([
        [1, 0.69, 0.92, 0, 0, 0],
        [-0.8, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.2, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.2, 0.1600, 0.4100, -0.22, 0, 18],
        [0.1, 0.2100, 0.2500, 0, 0.35, 0],
        [0.1, 0.0460, 0.0460, 0, 0.1, 0],
        [0.1, 0.0460, 0.0460, 0, -0.1, 0],
        [0.1, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.1, 0.0230, 0.0230, 0, -0.606, 0],
        [0.1, 0.0230, 0.0460, 0.06, -0.605, 0]
    ])

    # Normalized grid
    xn = (np.arange(0, N) - (N - 1) / 2) / ((N - 1) / 2)
    Xn, Yn = np.meshgrid(xn, xn)

    # Initialize the phantom
    X = np.zeros((N, N))

    # Add each ellipse to the phantom
    for i in range(e.shape[0]):
        a2 = e[i, 1]**2
        b2 = e[i, 2]**2
        x0 = e[i, 3]
        y0 = e[i, 4]
        phi = np.deg2rad(e[i, 5])
        A = e[i, 0]

        x = Xn - x0
        y = Yn - y0

        # Find indices inside the ellipse
        ellipse = ((x * np.cos(phi) + y * np.sin(phi))**2 / a2 +
                   (y * np.cos(phi) - x * np.sin(phi))**2 / b2 <= 1)

        # Add the intensity of the ellipse
        X[ellipse] += A

    # Flatten the phantom and ensure nonnegative values
    X = X.ravel()
    X[X < 0] = 0

    return X


def myphantom2(N):
    """
    Creates a circular phantom with radius N/2.

    This function generates a binary circular phantom with a discretization
    of N x N and returns it as a flattened vector.

    Parameters
    ----------
    N : int
        Number of discretization intervals in each dimension. The phantom is 
        represented by an N x N grid.

    Returns
    -------
    numpy.ndarray
        A flattened vector representing the circular phantom with N^2 elements.

    Notes
    -----
    - The circular phantom consists of a uniform circle centered at the grid 
      center with a radius of N/2.

    Original MATLAB code from AIR Tools, adapted to Python by Antony Vamvakeros.
    """
    X = np.zeros((N, N))
    radius2 = (N / 2)**2

    for i in range(N):
        for j in range(N):
            if ((i - N / 2)**2 + (j - N / 2)**2) <= radius2:
                X[i, j] = 1

    # Flatten the phantom
    return X.ravel()
	

def fstomo(N, omegas, rs):
    """
    Creates a sparse matrix for a 2D tomography problem using fan-beam geometry.

    This function generates the forward projection matrix `A` for a 2D domain of 
    size N x N. The rays are defined by their angular positions (`omegas`) and 
    their radial positions (`rs`). The matrix `A` can be used for tomographic 
    reconstruction or other simulation purposes.

    Parameters
    ----------
    N : int
        Number of discretization intervals in each dimension. The domain consists 
        of N x N cells.
    omegas : array-like
        Array of angles in radians, specifying the directions of the rays.
    rs : array-like
        Array of radial positions of the rays, specifying the starting positions 
        of each ray in the fan-beam geometry.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix `A` with shape (len(rs), N^2). Each row corresponds to a 
        ray, and each column corresponds to a cell in the domain.

    Notes
    -----
    - The matrix `A` represents the contribution of each ray to the cells it 
      intersects, with values corresponding to the length of the ray within each cell.
    - The function assumes that all rays are fully contained within the domain.

    References
    ----------
    Adapted from original MATLAB code in AIR Tools and modified by Antony Vamvakeros.
    """
    p = N  # Number of parallel rays (assumed equal to N for simplicity)
    w = N  # Width of the domain (assumed equal to N)

    nA = len(omegas)  # Number of angles
    npoints = len(rs)  # Number of rays

    # Initial coordinates for rays
    x0 = np.linspace(-w / 2, w / 2, p).reshape(-1, 1)
    y0 = np.zeros((p, 1))

    # Grid boundaries
    x = np.arange(-N / 2, N / 2 + 1)
    y = x

    # Initialize storage for sparse matrix construction
    rows = np.zeros(2 * N * nA)
    cols = np.zeros(2 * N * nA)
    vals = np.zeros(2 * N * nA)
    idxend = 0

    # Loop through each ray
    for point in tqdm(range(npoints), desc="Processing rays"):
        # Starting points for the current ray
        x0theta = np.cos(omegas[point]) * x0 - np.sin(omegas[point]) * y0
        y0theta = np.sin(omegas[point]) * x0 + np.cos(omegas[point]) * y0

        # Direction vector for the current ray
        a = -np.sin(omegas[point])
        b = np.cos(omegas[point])

        # Compute intersections with grid lines
        tx = (x - x0theta[int(rs[point])]) / a
        yx = b * tx + y0theta[int(rs[point])]
        ty = (y - y0theta[int(rs[point])]) / b
        xy = a * ty + x0theta[int(rs[point])]

        # Combine intersection times and coordinates
        t = np.concatenate([tx, ty])
        xxy = np.concatenate([x, xy])
        yxy = np.concatenate([yx, y])

        # Sort intersections by time
        I = np.argsort(t)
        t = t[I]
        xxy = xxy[I]
        yxy = yxy[I]

        # Filter out intersections outside the domain
        valid = (xxy >= -N / 2) & (xxy <= N / 2) & (yxy >= -N / 2) & (yxy <= N / 2)
        xxy = xxy[valid]
        yxy = yxy[valid]

        # Remove duplicate intersection points
        unique = (np.abs(np.diff(xxy)) > 1e-10) | (np.abs(np.diff(yxy)) > 1e-10)
        xxy = xxy[:-1][unique]
        yxy = yxy[:-1][unique]

        # Calculate ray lengths within each cell
        d = np.sqrt(np.diff(xxy)**2 + np.diff(yxy)**2)
        numvals = d.size

        # Store values in the sparse matrix
        if numvals > 0:
            xm = 0.5 * (xxy[:-1] + xxy[1:]) + N / 2
            ym = 0.5 * (yxy[:-1] + yxy[1:]) + N / 2

            col = np.floor(xm) * N + (N - np.floor(ym)) - 1
            row = point

            idxstart = idxend
            idxend = idxstart + numvals
            idx = np.arange(idxstart, idxend)

            rows[idx] = row
            cols[idx] = col.astype(int)
            vals[idx] = d

    # Truncate excess zeros
    rows = rows[:idxend]
    cols = cols[:idxend]
    vals = vals[:idxend]

    # Construct sparse matrix
    A = csr_matrix((vals, (rows, cols)), shape=(npoints, N**2), dtype=np.float32)

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

    Examples
    --------
    - To solve a system and visualize the reconstruction:
      >>> x = cgls(A, b, K=50, plot=True)
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
      
    
def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    """
    Concatenates two CSC (Compressed Sparse Column) matrices along the column axis.

    This function takes two CSC matrices (`matrix1` and `matrix2`) and concatenates
    them column-wise to produce a new CSC matrix.

    Parameters
    ----------
    matrix1 : scipy.sparse.csc_matrix
        The first CSC matrix to concatenate.
    matrix2 : scipy.sparse.csc_matrix
        The second CSC matrix to concatenate.

    Returns
    -------
    scipy.sparse.csc_matrix
        A new CSC matrix resulting from the column-wise concatenation of `matrix1` 
        and `matrix2`.

    Notes
    -----
    - Both input matrices must have the same number of rows. If they do not, 
      the function will raise a ValueError.
    - The resulting matrix is also in CSC format.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> import numpy as np
    >>> mat1 = csc_matrix(np.array([[1, 0], [0, 2]]))
    >>> mat2 = csc_matrix(np.array([[0, 3], [4, 0]]))
    >>> result = concatenate_csc_matrices_by_columns(mat1, mat2)
    >>> print(result.toarray())
    [[1 0 0 3]
     [0 2 4 0]]
    """
    # Check that the number of rows is the same
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError("Both matrices must have the same number of rows to concatenate.")

    # Concatenate the data and indices arrays
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))

    # Adjust the column pointers (indptr) for the second matrix
    offset = matrix1.indptr[-1]  # Offset for the data indices
    new_ind_ptr = np.concatenate((matrix1.indptr, matrix2.indptr[1:] + offset))

    # Create the new CSC matrix
    return csc_matrix((new_data, new_indices, new_ind_ptr), shape=(matrix1.shape[0], matrix1.shape[1] + matrix2.shape[1]))
	
	
def correct_sick_pixels_image(img, sick_pixels_cols, sick_pixels_rows):
    """
    Corrects "sick" pixels in an image by interpolating values from neighboring pixels.

    This function handles problematic pixels in an image (referred to as "sick" pixels) 
    based on their row and column indices. The correction is performed by replacing 
    the value of each "sick" pixel with the average value of its valid neighboring pixels.

    Parameters
    ----------
    img : numpy.ndarray
        Input 2D image array to be corrected.
    sick_pixels_cols : tuple
        Tuple of arrays specifying the row and column indices of "sick" pixels 
        identified from column differences. The format is `(row_indices, col_indices)`.
    sick_pixels_rows : tuple
        Tuple of arrays specifying the row and column indices of "sick" pixels 
        identified from row differences. The format is `(row_indices, col_indices)`.

    Returns
    -------
    numpy.ndarray
        Corrected image array.

    Notes
    -----
    - For border pixels, only valid neighbors within the image boundaries are used 
      for interpolation.
    - This function assumes that "sick" pixels are sparsely distributed and that 
      their neighbors are valid for interpolation.

    """
    # Handle "sick" pixels from column differences
    n_sick_cols = len(sick_pixels_cols[0])
    for ii in range(n_sick_cols):
        row = sick_pixels_cols[0][ii]       # Row index
        col = sick_pixels_cols[1][ii] + 1  # Adjust column index by +1 for `diff_cols`
        
        # Handle column border cases
        if col - 1 >= 0 and col + 1 < img.shape[1]:  # Not at the left or right border
            img[row, col] = np.mean([img[row, col - 1], img[row, col + 1]])
        elif col - 1 < 0:  # At the left border, take only the right neighbor
            img[row, col] = img[row, col + 1]
        elif col + 1 >= img.shape[1]:  # At the right border, take only the left neighbor
            img[row, col] = img[row, col - 1]

    # Handle "sick" pixels from row differences
    n_sick_rows = len(sick_pixels_rows[0])
    for ii in range(n_sick_rows):
        row = sick_pixels_rows[0][ii] + 1  # Adjust row index by +1 for `diff_rows`
        col = sick_pixels_rows[1][ii]     # Column index
        
        # Handle row border cases
        if row - 1 >= 0 and row + 1 < img.shape[0]:  # Not at the top or bottom border
            img[row, col] = np.mean([img[row - 1, col], img[row + 1, col]])
        elif row - 1 < 0:  # At the top border, take only the bottom neighbor
            img[row, col] = img[row + 1, col]
        elif row + 1 >= img.shape[0]:  # At the bottom border, take only the top neighbor
            img[row, col] = img[row - 1, col]

    return img

def ring_remover_post_recon_stripe(img, size=300, dim=1, **options):
    """
    Removes ring artifacts from a reconstructed image using stripe removal in polar coordinates.

    This function converts the input image from Cartesian to polar coordinates, 
    removes stripe artifacts in the polar domain, and maps the processed image back 
    to Cartesian coordinates. It is particularly useful for post-reconstruction 
    artifact correction in tomography.

    Parameters
    ----------
    img : numpy.ndarray
        Input 2D reconstructed image containing ring artifacts.
    size : int, optional, default=300
        Size of the filter used for stripe removal in the polar domain. This controls 
        the degree of smoothing and artifact suppression.
    dim : int, optional, default=1
        Dimension along which the stripe artifacts are removed in the polar domain:
        - 0: Vertical stripes.
        - 1: Horizontal stripes.
    **options : dict, optional
        Additional keyword arguments passed to `remo.remove_stripe_based_sorting`. These 
        can be used to customize the stripe removal process.

    Returns
    -------
    numpy.ndarray
        The processed image with reduced ring artifacts, mapped back to Cartesian coordinates.

    Notes
    -----
    - The function uses utility functions from the `algotom` library for coordinate 
      transformations and stripe removal.
    - Stripe removal in the polar domain effectively reduces ring artifacts in the 
      reconstructed Cartesian image.
    """
    (nrow, ncol) = img.shape
    (x_mat, y_mat) = util.rectangular_from_polar(ncol, ncol, ncol, ncol)
    (r_mat, theta_mat) = util.polar_from_rectangular(ncol, ncol, ncol, ncol)
    polar_mat = util.mapping(img, x_mat, y_mat)
    polar_mat = remo.remove_stripe_based_sorting(polar_mat, size=size, dim=dim,  **options)
    mat_rec = util.mapping(polar_mat, r_mat, theta_mat)
    return mat_rec