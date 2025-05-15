# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony Vamvakeros
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import algotom.util.utility as util
import algotom.prep.removal as remo
        

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
    
    if len(di) == 3:
        loop_range = tqdm(range(sinograms.shape[2])) if pbar else range(sinograms.shape[2])

    if colf is None:
        colf = sinograms.shape[1]  # Use all columns if colf is not specified
    
    if len(di) == 3:  # 3D sinograms

        if method == "both":
            air = (
                np.mean(sinograms[0:ofs, coli:colf, :], axis=(0, 1)) +
                np.mean(sinograms[-ofs:, coli:colf, :], axis=(0, 1))
            ) / 2
        elif method == "top":
            air = np.mean(sinograms[0:ofs, coli:colf, :], axis=(0, 1))
        elif method == "bottom":
            air = np.mean(sinograms[-ofs:, coli:colf, :], axis=(0, 1))

        for ii in loop_range:
            sinograms[:, :, ii] -= air[ii]

    elif len(di) == 2:  # 2D sinograms

        if method == "both":
            air = (
                np.mean(sinograms[0:ofs, coli:colf], axis=(0, 1)) +
                np.mean(sinograms[-ofs:, coli:colf], axis=(0, 1))
            ) / 2
        elif method == "top":
            air = np.mean(sinograms[0:ofs, coli:colf], axis=(0, 1))
        elif method == "bottom":
            air = np.mean(sinograms[-ofs:, coli:colf], axis=(0, 1))
            
        for ii in range(sinograms.shape[1]):

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


def sinocomcor(sinograms, interp=False, sine_wave=False, sine_wave_plot=False, pbar=False):
    """
    Corrects sinograms for motor jitter by aligning each projection based on its center of mass (COM).
    
    This function mitigates translation misalignments in sinograms (commonly due to motor jitter) by 
    shifting each projection so that its center of mass aligns with that of the first projection. The 
    method supports both 2D and 3D sinograms, where the third dimension (in the 3D case) typically 
    represents the spectral or z-axis.

    Optionally, the COM offsets can be fitted to a sine wave model, and this fitted model used for 
    correction instead of the raw COM offsets. This is useful in setups where jitter has a periodic 
    component (e.g. sinusoidal drift of a stage or motor).

    Parameters
    ----------
    sinograms : ndarray
        The input sinograms to correct. Can be:
        - A 2D array of shape (translations, projections), or
        - A 3D array of shape (translations, projections, z/spectral).
    
    interp : bool, optional
        Whether to use interpolation for shifts. If True, extrapolated values outside the original 
        data range are not clipped (default linear interpolation). If False, extrapolated values 
        are set to 0. Default is False.

    sine_wave : bool, optional
        If True, a sine wave will be fitted to the COM offsets, and the fitted curve used for 
        correction instead of the raw COM values. This assumes periodic jitter. Default is False.

    sine_wave_plot : bool, optional
        If True and `sine_wave` is enabled, the fitted sine curve will be plotted alongside the 
        raw COM offsets for visual inspection. Default is False.

    pbar : bool, optional
        If True, displays a progress bar during the correction process (useful for large datasets). 
        Default is False.

    Returns
    -------
    ndarray
        Sinograms with corrected projection alignment. Output has the same shape as the input.

    Notes
    -----
    - Center of mass is computed using `scipy.ndimage.center_of_mass` on each projection.
    - The first projection is used as the reference for alignment.
    - Corrections are performed via linear interpolation (`np.interp`) along the translation axis.
    - If `sine_wave` is True, the center of mass offset is modeled as:
          A * sin(2π f x + phi) + C
      and this model is fitted using `scipy.optimize.curve_fit`.
    - If using `interp=False`, values beyond the data range are filled with zeros (hard shift).

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
    com = com[:,0]
    
    if sine_wave:   
        def sine_model(x, A, f, phi, C):
            """
            Sine wave model: A * sin(2π f x + phi) + C
            """
            return A * np.sin(2 * np.pi * f * x + phi) + C
    
        xold = np.arange(ss.shape[0])
        
        # Initial guess for parameters: A, f, phi, C
        initial_guess = [1, 1, 0, 0]

        # Fit the model
        params, _ = curve_fit(sine_model, xold, com, p0=initial_guess)

        # Extract fitted parameters
        A_fit, f_fit, phi_fit, C_fit = params

        # Generate fitted curve
        y_fit = sine_model(xold, A_fit, f_fit, phi_fit, C_fit)

        if sine_wave_plot:
            
            plt.figure(1);plt.clf()
            plt.plot(xold, com, label="Observed", alpha=0.6)
            plt.plot(xold, y_fit, label="Fitted Sine", linewidth=2)
            plt.legend()
            plt.title(f"Fitted: A={A_fit:.2f}, f={f_fit:.2f}, phi={phi_fit:.2f}, C={C_fit:.2f}")
            plt.xlabel("x")

    # Create an empty array for corrected sinograms
    sn = np.zeros_like(sinograms)
    xold = np.arange(sn.shape[0])

    loop_range = tqdm(range(sinograms.shape[1])) if pbar else range(sinograms.shape[1])
    
    if len(di) == 2:  # For 2D sinograms
        for ii in loop_range:
            if sine_wave:
                xnew = xold - (y_fit[ii] - com[ii])
            else:
                xnew = xold + com[ii]
            if interp:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii])
            else:
                sn[:, ii] = np.interp(xnew, xold, sinograms[:, ii], left=0, right=0)

    elif len(di) == 3:  # For 3D sinograms
        for ll in loop_range:
            for ii in range(sinograms.shape[1]):
                if sine_wave:
                    xnew = xold - (y_fit[ii] - com[ii])
                else:
                    xnew = xold + com[ii]
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