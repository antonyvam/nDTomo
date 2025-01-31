# -*- coding: utf-8 -*-
"""
Misc3D  tools for nDTomo

@author: Antony Vamvakeros
"""

from numpy import max, linspace, histogram
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
import matplotlib.pyplot as plt

def cmap_to_ctf(cmap_name, vmin=0, vmax=1):
    """
    Converts a Matplotlib colormap into a Mayavi ColorTransferFunction.

    Parameters
    ----------
    cmap_name : str
        The name of the Matplotlib colormap to be converted.
    vmin : float, optional (default=0)
        The minimum value for the colormap range.
    vmax : float, optional (default=1)
        The maximum value for the colormap range.

    Returns
    -------
    ColorTransferFunction
        A Mayavi ColorTransferFunction representing the specified colormap.
    """
    values = linspace(vmin, vmax, 256)
    cmap = plt.colormaps.get_cmap(cmap_name)(values)  # Updated for Matplotlib 3.7+
    ctf = ColorTransferFunction()
    for i, v in enumerate(values):
        ctf.add_rgb_point(v, cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return ctf
	

def create_adaptive_opacity_function(vol, vmin, vmax):
    """
    Creates an adaptive Opacity Transfer Function based on the volume's intensity distribution.

    Parameters
    ----------
    vol : np.ndarray
        3D volume data used to compute intensity distribution.
    vmin : float
        The minimum intensity value in the volume.
    vmax : float
        The maximum intensity value in the volume.

    Returns
    -------
    PiecewiseFunction
        A Mayavi PiecewiseFunction defining the opacity transfer function.
    """
    otf = PiecewiseFunction()
    
    # Compute histogram to understand intensity distribution
    hist, bin_edges = histogram(vol, bins=10, range=(vmin, vmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Define opacity dynamically based on intensity distribution
    for i, bin_value in enumerate(bin_centers):
        opacity = min(0.2 + hist[i] * 5, 1.0)  # Scale by histogram density
        otf.add_point(bin_value, opacity)

    return otf
	
	
def create_balanced_opacity_function(vmin, vmax):
    """
    Creates a balanced Opacity Transfer Function to provide smooth transparency transitions.

    Parameters
    ----------
    vmin : float
        The minimum intensity value for opacity scaling.
    vmax : float
        The maximum intensity value for opacity scaling.

    Returns
    -------
    PiecewiseFunction
        A Mayavi PiecewiseFunction with balanced opacity values.
    """
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.05)   # Almost transparent at min value
    otf.add_point(vmin + (vmax - vmin) * 0.25, 0.2)
    otf.add_point(vmin + (vmax - vmin) * 0.5, 0.4)
    otf.add_point(vmin + (vmax - vmin) * 0.75, 0.7)
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    
    return otf


def create_solid_opacity_function():
    """
    Creates an Opacity Transfer Function that makes all values fully opaque.

    Returns
    -------
    PiecewiseFunction
        A Mayavi PiecewiseFunction where all intensities are fully opaque.
    """
    otf = PiecewiseFunction()
    otf.add_point(0.0, 1.0)  # Fully opaque at minimum value
    otf.add_point(1.0, 1.0)  # Fully opaque at maximum value
    return otf

def create_fade_opacity_function(vmin, vmax):
    """
    Creates an Opacity Transfer Function that smoothly fades low-intensity values.

    Parameters
    ----------
    vmin : float
        The minimum intensity value for the opacity function.
    vmax : float
        The maximum intensity value for the opacity function.

    Returns
    -------
    PiecewiseFunction
        A Mayavi PiecewiseFunction where low values gradually fade in opacity.
    """
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.0)   # Fully transparent at zero
    otf.add_point(vmin + (vmax - vmin) * 0.05, 0.2)  # Slightly visible
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.6)  # More visible
    otf.add_point(vmax, 1.0)   # Fully opaque at max intensity
    
    return otf

def create_binary_opacity_function(threshold):
    """
    Creates an Opacity Transfer Function with a binary threshold.

    Parameters
    ----------
    threshold : float
        The intensity threshold where values below are fully transparent 
        and values above are fully opaque.

    Returns
    -------
    PiecewiseFunction
        A Mayavi PiecewiseFunction implementing a binary opacity step.
    """
    otf = PiecewiseFunction()

    otf.add_point(threshold - 1e-6, 0.0)  # Just before the threshold → fully transparent
    otf.add_point(threshold, 1.0)  # At threshold → fully opaque
    otf.add_point(1.0, 1.0)  # Everything else remains fully visible

    return otf
	
def showvol(vol, vlim=None, colormap="jet", show_axes=True, show_colorbar=True,
			interp = 'linear', opacity_mode = 'adaptive', thr = 0.05):
    """
    Performs volume rendering of a 3D dataset using Mayavi's mlab with various customization options.

    Parameters
    ----------
    vol : np.ndarray
        3D volume data to be visualized.
    vlim : tuple, optional
        (vmin, vmax) for intensity scaling. If None, it defaults to (0, max(vol)).
    colormap : str, optional
        The colormap used for visualization (e.g., "jet", "viridis", "gray").
    show_axes : bool, optional
        Whether to display the coordinate axes in the visualization. Default is True.
    show_colorbar : bool, optional
        Whether to display a colorbar corresponding to intensity values. Default is True.
    interp : str, optional
        The interpolation type for rendering. Available options:
        - "linear" (default)
        - "nearest"
        - "cubic"
    opacity_mode : str, optional
        The opacity transfer function mode. Options include:
        - 'binary'   : Applies a binary threshold-based opacity function.
        - 'fade'     : Applies a smoothly fading opacity function.
        - 'adaptive' : Uses an adaptive opacity function based on volume statistics.
        - 'solid'    : Renders the volume as fully opaque.
        Default is 'adaptive'.
    thr : float, optional
        The threshold value for the binary opacity mode. Values below `thr` are 
        rendered as transparent, while values above are fully opaque. Default is 0.05.

    Returns
    -------
    None
        The function renders the volume but does not return any objects.
    """
	
    if vlim is None:
        vmin = 0
        vmax = max(vol)
    else:
        vmin, vmax = vlim
    
    # Ensure the figure is managed by mlab
    fig = mlab.gcf()  # Get the current figure

    # Create volume rendering explicitly linked to the figure
    src = mlab.pipeline.scalar_field(vol, figure=fig)
    volume = mlab.pipeline.volume(src, vmin=vmin, vmax=vmax, figure=fig)
    volume._volume_property.interpolation_type = interp
	
    # Convert colormap to ColorTransferFunction and apply it
    ctf = cmap_to_ctf(colormap, vmin, vmax)
    volume._volume_property.set_color(ctf)
    volume._ctf = ctf
    volume.update_ctf = True

    # # Apply Adaptive Opacity Transfer Function
    if opacity_mode == 'binary':
        otf = create_binary_opacity_function(thr)
    elif opacity_mode == 'fade':
        otf = create_fade_opacity_function(vmin, vmax)
    elif opacity_mode == 'adaptive':
        otf = create_adaptive_opacity_function(vol, vmin, vmax)
    elif opacity_mode == 'solid':
        otf = create_solid_opacity_function()
		
    volume._volume_property.set_scalar_opacity(otf)
    volume._otf = otf
    volume.update_ctf = True 

    # Extract and apply the same LUT to the colorbar
    lut_manager = volume.module_manager.scalar_lut_manager
    lut_manager.lut_mode = colormap  # Ensure the colorbar follows the colormap

    # Show colorbar
    if show_colorbar:
        mlab.colorbar(orientation="vertical", title="Intensity")

    # Toggle axes visibility
    if show_axes:
        mlab.orientation_axes()
