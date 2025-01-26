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
    
    '''
    Custom colormap: It is the jet colormap but 0 correponds to black
    '''
    
    jet = cm.get_cmap('jet', 256)
    newcolors = jet(linspace(0, 1, 256))
    black = array([0, 0, 0, 1])
    newcolors[0, :] = black
    newcmp = ListedColormap(newcolors)
    return(newcmp)

def showplot(spectrum, fignum = 1):
    
    plt.figure(fignum);plt.clf()
    plt.plot(spectrum)
    plt.show()
    
def showim(im, fignum = 1, clim=None, cmap='jet'):
    
    plt.figure(fignum);plt.clf()
    plt.imshow(im, cmap = cmap)
    plt.colorbar()
    plt.axis('tight')
    if clim is None:
        plt.clim(min(im), max(im))
    else:
        plt.clim(clim)
    plt.show()
    
def showspectra(spectra, fignum = 1):
    
    plt.figure(fignum);plt.clf()
    for ii in range(len(spectra)):
        plt.plot(spectra[ii])
    plt.show()
       
def closefigs():
    
    plt.close('all')
    
def plotfigs_imgs(imagelist, legendlist=None, rows=1, cols=5, figsize=(20,3), cl=True, cmap = 'jet'):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
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

    imagelist = []; legendlist = []
    for ii in range(components.shape[0]):
        im = components[ii,:]
        imagelist.append(reshape(im, (xpix, ypix)))
        legendlist.append('Component %d' %(ii+1))
        
    return(imagelist, legendlist)


    
def plotfigs_spectra(spectralist, legendlist= None, xaxis=None, rows=1, cols=5, figsize=(20,3)):
    
    '''
    Create a collage of images without xticks/yticks
    
    @author: Antony Vamvakeros and Thanasis Giokaris
    '''
    
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

    splist = []; legendlist = []
    for ii in range(components.shape[0]):
        splist.append(components[ii,:])
        legendlist.append('Component %d' %(ii+1))
        
    return(splist, legendlist)