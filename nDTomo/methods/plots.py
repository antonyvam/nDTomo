# -*- coding: utf-8 -*-
"""
Methods related for visual inspection of data

@author: Antony Vamvakeros
"""

from numpy import arange, array, linspace, min, max, reshape
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def nDTomo_colormap():
    
    """
    Create a custom colormap based on 'jet' with zero value mapped to black.

    Returns
    -------
    matplotlib.colors.ListedColormap
        Modified jet colormap with the first color (zero value) set to black.
    """
    
    jet = cm.get_cmap('jet', 256)
    newcolors = jet(linspace(0, 1, 256))
    black = array([0, 0, 0, 1])
    newcolors[0, :] = black
    newcmp = ListedColormap(newcolors)
    return(newcmp)

def showplot(spectrum, fignum = 1):

    """
    Display a 1D spectrum using matplotlib.

    Parameters
    ----------
    spectrum : array-like
        1D array containing the spectrum to be plotted.
    fignum : int, optional
        Figure number for the matplotlib window (default is 1).
    """
        
    plt.figure(fignum);plt.clf()
    plt.plot(spectrum)
    plt.show()
    
def showim(im, fignum = 1, clim=None, cmap='jet'):
    
    """
    Display a 2D image using matplotlib with optional colormap and color limits.

    Parameters
    ----------
    im : array-like
        2D array representing the image to display.
    fignum : int, optional
        Figure number for the matplotlib window (default is 1).
    clim : tuple of float, optional
        Tuple specifying the color limits (vmin, vmax). If None, uses data min/max.
    cmap : str or Colormap, optional
        Colormap to use for image display (default is 'jet').
    """
    plt.figure(fignum);plt.clf()
    plt.imshow(im, cmap = cmap)
    plt.colorbar()
    plt.axis('tight')
    if clim is None:
        plt.clim(min(im), max(im))
    else:
        plt.clim(clim)
    plt.show()
    

def showspectra(spectra, labels=None, fig_num=1):
    """
    Display multiple spectra on a single plot.

    Parameters
    ----------
    spectra : list of np.ndarray
        List of 1D arrays representing the spectra to be plotted.
    labels : list of str, optional
        Labels for each spectrum. If None, no legend is shown.
    fig_num : int, optional
        Figure number to use for the plot.
    """
    plt.figure(fig_num)
    plt.clf()

    for i, spec in enumerate(spectra):
        if labels is not None and i < len(labels):
            plt.plot(spec, label=labels[i])
        else:
            plt.plot(spec)

    if labels is not None:
        plt.legend()
    plt.xlabel('Channel')
    plt.ylabel('Intensity')
    plt.title('Spectral Overlay')
    plt.tight_layout()
    plt.show()
       
def closefigs():
    """
    Close all open matplotlib figures.
    """    
    plt.close('all')
    
def plotfigs_imgs(imagelist, legendlist=None, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'jet'):
    
    """
    Plot a grid of 2D images with optional legends, colorbars, and custom layout.

    Parameters
    ----------
    imagelist : list of ndarray
        List of 2D arrays to display as images.
    legendlist : list of str, optional
        List of titles for each subplot. If None, default labels like 'Component 1' will be used.
    rows : int, optional
        Number of subplot rows (default is 1).
    cols : int, optional
        Number of subplot columns (default is 5).
    figsize : tuple, optional
        Size of the entire figure in inches (default is (20, 3)).
    cl : bool, optional
        If True, display colorbar for each subplot (default is True).
    cmap : str or Colormap, optional
        Colormap used for displaying the images (default is 'jet').
    """
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(imagelist)):

            legendlist.append('Component %d' %(ii+1))        
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(imagelist)):
            
            i = axes[ii].imshow(imagelist[ii], cmap=cmap)
            axes[ii].set_axis_off()
            axes[ii].set_title(legendlist[ii])
    
            if cl==True:
                fig.colorbar(i, ax=axes[ii])

    elif len(axes.shape)==2:
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                print(kk)
                
                if kk < len(imagelist):
            
                    i = axes[ii,jj].imshow(imagelist[kk], cmap=cmap)
                    axes[ii,jj].set_axis_off()
                    axes[ii,jj].set_title(legendlist[kk])
            
                    if cl==True:
                        fig.colorbar(i, ax=axes[ii,jj])        
                    
                    kk = kk + 1

def create_complist_imgs(components, xpix, ypix):     

    """
    Convert a 2D array of flattened components into a list of reshaped images and legend titles.

    Parameters
    ----------
    components : ndarray
        2D array where each row is a flattened image (n_components x n_pixels).
    xpix : int
        Width of each image in pixels.
    ypix : int
        Height of each image in pixels.

    Returns
    -------
    tuple
        A tuple (imagelist, legendlist) where:
        - imagelist is a list of 2D arrays shaped as (xpix, ypix).
        - legendlist is a list of string labels like 'Component 1', 'Component 2', etc.
    """
    
    imagelist = []; legendlist = []
    for ii in range(components.shape[0]):
        im = components[ii,:]
        imagelist.append(reshape(im, (xpix, ypix)))
        legendlist.append('Component %d' %(ii+1))
        
    return(imagelist, legendlist)


    
def plotfigs_spectra(spectralist, legendlist= None, xaxis=None, rows=1, cols=5, figsize=(20,3)):
    
    """
    Plot a grid of spectra in a subplot layout with optional legends and custom x-axis.

    Parameters
    ----------
    spectralist : list of ndarray
        List of 1D arrays representing spectra.
    legendlist : list of str, optional
        Titles for each subplot. If None, default labels like 'Component 1' will be used.
    xaxis : ndarray, optional
        Shared x-axis for all spectra. If None, the index of the spectrum is used.
    rows : int, optional
        Number of subplot rows (default is 1).
    cols : int, optional
        Number of subplot columns (default is 5).
    figsize : tuple, optional
        Size of the figure in inches (default is (20, 3)).
    """
    
    if legendlist is None:
        
        legendlist = []
        
        for ii in range(len(legendlist)):

            legendlist.append('Component %d' %(ii+1))    
    
    if xaxis is None:
        
        xaxis = arange(0, len(spectralist[0]))
            
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if len(axes.shape)<2:
        for ii in range(len(spectralist)):
            
            axes[ii].plot(xaxis, spectralist[ii])
            axes[ii].set_title(legendlist[ii])

    elif len(axes.shape)==2:
        
        kk = 0
        for ii in range(axes.shape[0]):
            for jj in range(axes.shape[1]):
            
                print(kk)
                
                if kk < len(spectralist):
            
                    axes[ii,jj].plot(xaxis, spectralist[kk])
                    axes[ii,jj].set_title(legendlist[kk])
   
                    kk = kk + 1
                    
def create_complist_spectra(components):     

    """
    Convert a 2D array of spectral components into a list of 1D spectra and corresponding labels.

    Parameters
    ----------
    components : ndarray
        2D array where each row is a spectral component.

    Returns
    -------
    tuple
        A tuple (splist, legendlist) where:
        - splist is a list of 1D spectral arrays.
        - legendlist is a list of string labels like 'Component 1', 'Component 2', etc.
    """
    
    splist = []; legendlist = []
    for ii in range(components.shape[0]):
        splist.append(components[ii,:])
        legendlist.append('Component %d' %(ii+1))
        
    return(splist, legendlist)