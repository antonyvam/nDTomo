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
    """Convert a Matplotlib colormap to a Mayavi ColorTransferFunction."""
    values = linspace(vmin, vmax, 256)
    cmap = plt.colormaps.get_cmap(cmap_name)(values)  # Updated for Matplotlib 3.7+
    ctf = ColorTransferFunction()
    for i, v in enumerate(values):
        ctf.add_rgb_point(v, cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return ctf
	

def create_adaptive_opacity_function(vol, vmin, vmax):
    """Create an adaptive Opacity Transfer Function based on volume intensity distribution."""
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
    """Create a more uniform Opacity Transfer Function to prevent full transparency in regions."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.05)   # Almost transparent at min value
    otf.add_point(vmin + (vmax - vmin) * 0.25, 0.2)
    otf.add_point(vmin + (vmax - vmin) * 0.5, 0.4)
    otf.add_point(vmin + (vmax - vmin) * 0.75, 0.7)
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    
    return otf


def create_solid_opacity_function():
    """Create an Opacity Transfer Function that makes everything solid (fully opaque)."""
    otf = PiecewiseFunction()
    otf.add_point(0.0, 1.0)  # Fully opaque at minimum value
    otf.add_point(1.0, 1.0)  # Fully opaque at maximum value
    return otf

def create_fade_opacity_function(vmin, vmax):
    """Smoothly fade low-intensity values instead of hard transparency."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.0)   # Fully transparent at zero
    otf.add_point(vmin + (vmax - vmin) * 0.05, 0.2)  # Slightly visible
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.6)  # More visible
    otf.add_point(vmax, 1.0)   # Fully opaque at max intensity
    
    return otf

def create_binary_opacity_function(threshold):
    """Creates an opacity function where values below `threshold` are transparent, 
    and values above it are fully opaque."""
    otf = PiecewiseFunction()

    otf.add_point(threshold - 1e-6, 0.0)  # Just before the threshold → fully transparent
    otf.add_point(threshold, 1.0)  # At threshold → fully opaque
    otf.add_point(1.0, 1.0)  # Everything else remains fully visible

    return otf
	
def showvol(vol, vlim=None, colormap="jet", show_axes=True, show_colorbar=True,
			interp = 'linear', opacity_mode = 'adaptive', thr = 0.05):
    '''
    Volume rendering using Mayavi mlab with customization options.
    
    Parameters:
        vol (np.ndarray): 3D volume data.
        vlim (tuple): (vmin, vmax) for intensity scaling. Default is (0, max(vol)).
        colormap (str): Colormap to use (e.g., "jet", "viridis", "gray").
        show_axes (bool): Whether to display the coordinate axes. Default is True.
        show_colorbar (bool): Whether to show a colorbar. Default is True.
        interp (str): Interpolation type for rendering. Options are "linear", 
                      "nearest", or "cubic". Default is "linear".
        opacity_mode (str): Mode for opacity transfer function. Options are:
            - 'binary': Apply a binary threshold-based opacity function.
            - 'fade': Apply a smooth fading opacity function.
            - 'adaptive': Apply an adaptive opacity function based on volume statistics.
            - 'solid': Make the entire volume fully opaque.
            Default is 'adaptive'.
        thr (float): Threshold for the binary opacity mode. Values below `thr` 
                     are transparent, and values above `thr` are fully opaque. 
                     Default is 0.05.
    '''
	
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
    else:
        mlab.axes(visible=False)