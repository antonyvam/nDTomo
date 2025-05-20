# -*- coding: utf-8 -*-
"""
Neural networks models

@author: Antony Vamvakeros
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PeakFitCNN(nn.Module):
    
    """
    A 2D convolutional neural network designed for upsampling and refining spectral or spatial peak data,
    optionally doubling or quadrupling the input resolution using bilinear interpolation and CNN blocks.

    Parameters
    ----------
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    nfilts : int
        Number of filters in the intermediate convolution layers.
    upscale_factor : int
        Upscaling factor for the input. Supported values: 2 or 4.
    norm_type : str
        Type of normalization to apply: 'instance', 'batch', or 'layer'.
    activation : str
        Final activation function: 'Linear', 'ReLU', 'Sigmoid', or 'LeakyReLU'.
    padding : str
        Padding mode for convolutions ('same' or 'valid').
    npix : int
        Number of pixels in the input (required for LayerNorm).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (batch_size, nch_in, H, W).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nch_out, H * upscale_factor, W * upscale_factor).
    """
        
    def __init__(self, nch_in=1, nch_out=1, nfilts=32, upscale_factor = 4,
                 norm_type='instance', activation='Linear', padding='same', npix=None):
        super(PeakFitCNN, self).__init__()

        self.npix = npix
        self.upscale_factor = upscale_factor
        # Initial feature extraction
        self.input = nn.Conv2d(nch_in, nfilts, kernel_size=3, stride=1, padding=padding, bias=True)

        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding=padding, bias=True))
        # Add normalization based on norm_type
        if norm_type == "instance":
            layers.append(nn.InstanceNorm2d(nfilts, affine=True))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm2d(nfilts))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm([nfilts, 2*self.npix, 2*self.npix]))

        # Add activation function
        layers.append(nn.ReLU(inplace=True))

        self.upsample1 = nn.Sequential(*layers)

        if self.upscale_factor == 4:
            layers = []
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding=padding, bias=True))
            # Add normalization based on norm_type
            if norm_type == "instance":
                layers.append(nn.InstanceNorm2d(nfilts, affine=True))
            elif norm_type == "batch":
                layers.append(nn.BatchNorm2d(nfilts))
            elif norm_type == "layer":
                layers.append(nn.LayerNorm([nfilts, 4*self.npix, 4*self.npix]))
            # Add activation function
            layers.append(nn.ReLU(inplace=True))

            self.upsample2 = nn.Sequential(*layers)

        # Final output layer
        self.xrdct = nn.Conv2d(nfilts, nch_out, kernel_size=3, stride=1, padding=padding, bias=True)

        # Final activation
        self.final_activation = None
        if activation == "ReLU":
            self.final_activation = nn.ReLU()
        elif activation == "Sigmoid":
            self.final_activation = nn.Sigmoid()
        elif activation == "LeakyReLU":
            self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):  # Feature maps from autoencoder2D are passed

        x = self.input(x)

        # Upsampling 1
        x = self.upsample1(x)

        if self.upscale_factor == 4:
            # Upsampling 2
            x = self.upsample2(x)

        # Output layer
        x = self.xrdct(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
    
class PrmCNN2D(nn.Module):
    
    """
    A flexible 2D model that combines a trainable tensor (image parameterization) with an optional CNN-based
    processing module. Can operate in three modes:
    - Pure parameterization (learned image).
    - CNN only (applies CNN to input).
    - Parameterization + CNN (CNN applied to learned image).

    Parameters
    ----------
    npix : int
        Image resolution (assumes square images).
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    nfilts : int
        Number of filters in CNN layers.
    nlayers : int
        Number of intermediate CNN blocks (excluding first and last layers).
    norm_type : str
        Type of normalization: 'layer', 'instance', or 'batchnorm'.
    prms_layer : bool
        If True, a learnable tensor is used as input.
    cnn_layer : bool
        If True, a CNN processes the input or parameter tensor.
    tensor_vals : str
        Initialization mode for the learned tensor: 'random', 'zeros', 'ones', 'mean', 'random_positive', or 'custom'.
    tensor_initial : torch.Tensor or None
        Custom tensor to use if tensor_vals == 'custom'.
    padding : str
        Padding mode for convolutions ('same' or 'valid').

    Forward
    -------
    x : torch.Tensor
        Input tensor if cnn_layer=True and prms_layer=False. Ignored otherwise.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (1, nch_out, npix, npix).
    """
        
    def __init__(self, npix, nch_in=1, nch_out=1, nfilts=32, nlayers=4, norm_type='layer', 
                 prms_layer=True, cnn_layer=True, tensor_vals = 'random', tensor_initial = None,
                 padding='same'):
        super(PrmCNN2D, self).__init__()
        self.npix = npix
        self.prms_layer = prms_layer
        self.cnn_layer = cnn_layer

        if self.prms_layer:
            if tensor_vals == 'random':
                self.initial_tensor = nn.Parameter(2*torch.randn(1, nch_in, npix, npix)-1)
            elif tensor_vals == 'zeros':
                self.initial_tensor = nn.Parameter(torch.zeros(1, nch_in, npix, npix))
            elif tensor_vals == 'ones':
                self.initial_tensor = nn.Parameter(torch.ones(1, nch_in, npix, npix))
            elif tensor_vals == 'mean':
                self.initial_tensor = nn.Parameter(0.5*torch.ones(1, nch_in, npix, npix))
            elif tensor_vals == 'random_positive':
                self.initial_tensor = nn.Parameter(torch.randn(1, nch_in, npix, npix))
            elif tensor_vals == 'custom':
                try:
                    self.initial_tensor = nn.Parameter(tensor_initial)
                except:
                    print('Custom tensor not provided. Using random tensor instead')
                    self.initial_tensor = nn.Parameter(torch.randn(1, nch_in, npix, npix))
        if self.cnn_layer:
            layers = []
            layers.append(nn.Conv2d(nch_in, nfilts, kernel_size=3, stride=1, padding=padding))  # 'same' padding in PyTorch is usually done by manually specifying the padding
            if norm_type=='layer':
                if padding=='valid':
                    layers.append(nn.LayerNorm([nfilts, self.npix -2, self.npix -2]))
                else:
                    layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))
            elif norm_type=='instance':
                layers.append(nn.InstanceNorm2d(nfilts, affine = True))
            elif norm_type=='batchnorm':            
                layers.append(nn.BatchNorm2d(nfilts))

            layers.append(nn.ReLU())

            for layer in range(nlayers):
                
                layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding=padding))
                if norm_type=='layer':
                    if padding=='valid':
                        layers.append(nn.LayerNorm([nfilts, self.npix -2*(layer + 2), self.npix -2*(layer + 2)]))
                    else:
                        layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))
                elif norm_type=='instance':
                    layers.append(nn.InstanceNorm2d(nfilts, affine = True))            
                elif norm_type=='batchnorm':            
                    layers.append(nn.BatchNorm2d(nfilts))

                layers.append(nn.ReLU())

            layers.append(nn.Conv2d(nfilts, nch_out, kernel_size=3, stride=1, padding=padding))
            layers.append(nn.Sigmoid())
            self.cnn2d = nn.Sequential(*layers)

    def forward(self, x):
        if self.prms_layer and self.cnn_layer:
            out = self.cnn2d(torch.sigmoid(self.initial_tensor))
        elif self.cnn_layer and not self.prms_layer:
            out = self.cnn2d(x)
        elif self.prms_layer and not self.cnn_layer:
            out = torch.sigmoid(self.initial_tensor)
        return out



class CNN1D(nn.Module):
    
    """
    A 1D convolutional neural network for sequential or spectral data processing with optional normalization 
    and residual connections.

    Parameters
    ----------
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    nfilts : int
        Number of filters in the convolutional layers.
    nlayers : int
        Number of intermediate convolutional blocks.
    norm_type : str or None
        Type of normalization: 'batch', 'layer', or None.
    activation : str
        Final activation type. If 'Sigmoid', a Sigmoid activation is appended after the last layer.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (batch_size, nch_in, sequence_length).
    residual : bool
        If True, adds input x to the output (residual connection).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nch_out, sequence_length).
    """
        
    def __init__(self, nch_in=1, nch_out=1, nfilts=32, nlayers=4, norm_type='layer', activation='Linear'):
        
        super(CNN1D, self).__init__()
        
        layers = []
        layers.append(nn.Conv1d(nch_in, nfilts, kernel_size=3, stride=1, padding=1))
        if norm_type is not None:
            self.add_norm_layer(layers, nfilts, norm_type)
        layers.append(nn.ReLU())
        
        for layer in range(nlayers):
            layers.append(nn.Conv1d(nfilts, nfilts, kernel_size=3, stride=1, padding=1))
            if norm_type is not None:
                self.add_norm_layer(layers, nfilts, norm_type)
            layers.append(nn.ReLU())

        layers.append(nn.Conv1d(nfilts, nch_out, kernel_size=3, stride=1, padding=1))
        
        if activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        
        self.cnn1d = nn.Sequential(*layers)

    def add_norm_layer(self, layers, nfilts, norm_type):
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(nfilts))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([nfilts]))
        else:
            raise ValueError('Invalid normalization type')
            
    def forward(self, x, residual=False):
        if residual:
            out = self.cnn1d(x) + x
        else:
            out = self.cnn1d(x)
        return out
    
class CNN2D(nn.Module):
    """
    A configurable 2D Convolutional Neural Network (CNN) for image processing tasks.

    Parameters
    ----------
    npix : int
        The spatial size (height and width) of the input images.
    nch_in : int, optional, default=1
        Number of input channels (e.g., 1 for grayscale images, 3 for RGB).
    nch_out : int, optional, default=1
        Number of output channels.
    nfilts : int, optional, default=32
        Number of filters (channels) in the intermediate convolutional layers.
    nlayers : int, optional, default=4
        Number of intermediate convolutional layers.
    norm_type : str or None, optional, default='layer'
        Normalization type to apply after convolutions:
        - 'batch': Batch normalization.
        - 'layer': Layer normalization.
        - None: No normalization.
    activation : str, optional, default='Linear'
        Final activation function to apply:
        - 'Sigmoid': Applies a sigmoid activation.
        - 'Linear': No activation applied.

    Methods
    -------
    forward(x, residual=False):
        Performs a forward pass through the network.
        - `residual`: If True, adds the input `x` to the output.

    """
    def __init__(self, npix, nch_in=1, nch_out=1, nfilts=32, nlayers=4, norm_type='layer', activation='Linear'):
        super(CNN2D, self).__init__()

        self.npix = npix
        layers = []

        # Input convolutional layer
        layers.append(nn.Conv2d(nch_in, nfilts, kernel_size=3, stride=1, padding=1))  # Same padding
        if norm_type:
            self.add_norm_layer(layers, nfilts, norm_type)
        layers.append(nn.ReLU())

        # Intermediate convolutional layers
        for _ in range(nlayers):
            layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding=1))
            if norm_type:
                self.add_norm_layer(layers, nfilts, norm_type)
            layers.append(nn.ReLU())

        # Output convolutional layer
        layers.append(nn.Conv2d(nfilts, nch_out, kernel_size=3, stride=1, padding=1))
        if activation == 'Sigmoid':
            layers.append(nn.Sigmoid())

        # Combine all layers into a sequential model
        self.cnn2d = nn.Sequential(*layers)

    def add_norm_layer(self, layers, nfilts, norm_type):
        """
        Adds a normalization layer to the list of layers.

        Parameters
        ----------
        layers : list
            List of layers to which the normalization layer is appended.
        nfilts : int
            Number of channels for normalization.
        norm_type : str
            Type of normalization ('batch' or 'layer').
        """
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(nfilts))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}. Choose 'batch', 'layer', or None.")

    def forward(self, x, residual=False):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nch_in, npix, npix).
        residual : bool, optional, default=False
            If True, adds the input `x` to the output.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, nch_out, npix, npix).
        """
        if residual:
            out = self.cnn2d(x) + x
        else:
            out = self.cnn2d(x)
        return out


class CNN3D(nn.Module):
    
    """
    A 3D convolutional neural network for volumetric data processing with optional normalization and 
    configurable depth.

    Parameters
    ----------
    npix : int
        Size of the 3D cube (assumes cube of shape npix x npix x npix).
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    nfilts : int
        Number of filters in convolutional layers.
    nlayers : int
        Number of intermediate convolutional blocks.
    norm_type : str or None
        Type of normalization: 'batch', 'layer', or None.
    activation : str
        Final activation type. If 'Sigmoid', a Sigmoid activation is appended.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (batch_size, nch_in, D, H, W).
    residual : bool
        If True, adds the input x to the output (residual connection).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nch_out, D, H, W).
    """    
    
    def __init__(self, npix, nch_in=1, nch_out=1, nfilts=32, nlayers=4, norm_type='layer', activation='Linear'):
        
        super(CNN3D, self).__init__()
        
        self.npix = npix
        
        layers = []
        layers.append(nn.Conv3d(nch_in, nfilts, kernel_size=3, stride=1, padding=1))  # 'same' padding in PyTorch is usually done by manually specifying the padding
        if norm_type is not None:
            self.add_norm_layer(layers, nfilts, norm_type)
        layers.append(nn.ReLU())
        
        for layer in range(nlayers):
            layers.append(nn.Conv3d(nfilts, nfilts, kernel_size=3, stride=1, padding=1))
            if norm_type is not None:
                self.add_norm_layer(layers, nfilts, norm_type)
            layers.append(nn.ReLU())

        layers.append(nn.Conv3d(nfilts, nch_out, kernel_size=3, stride=1, padding=1))
        
        if activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        
        self.cnn3d = nn.Sequential(*layers)

    def add_norm_layer(self, layers, nfilts, norm_type):
        if norm_type == 'batch':
            layers.append(nn.BatchNorm3d(nfilts))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([nfilts, self.npix, self.npix, self.npix]))
        else:
            raise ValueError('Invalid normalization type')
            
    def forward(self, x, residual=False):
        if residual:
            out = self.cnn3d(x) + x
        else:
            out = self.cnn3d(x)
        return out
    
class SD2I(nn.Module):
    
    """
    The SD2I model combines dense layers and 2D convolutions to generate high-resolution 2D images from a single input.
    It uses progressive upsampling and deep feature refinement and is suitable for inverse problems or image synthesis.

    Parameters
    ----------
    npix : int
        Target image size (npix x npix).
    factor : int
        Latent channel factor to expand dense output into feature maps.
    nims : int
        Number of output images (channels).
    nfilts : int
        Number of filters in convolutional blocks.
    ndense : int
        Width of fully connected layers before reshaping into feature maps.
    dropout : bool
        Whether to use dropout in dense layers.
    norm_type : str or None
        Normalization type: 'batch', 'layer', or None.
    upsampling : int
        Number of upsampling steps (1, 2, or 4).
    act_layer : str
        Final activation function: 'Sigmoid' or None.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (batch_size, 1) — a single scalar per sample.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nims, npix, npix).
    """
    
    def __init__(self, npix, factor=8, nims=5, nfilts = 32, ndense = 64, dropout=True, norm_type='layer', upsampling = 4, act_layer='Sigmoid'):
        
        super(SD2I, self).__init__()
        
        self.npix = npix
        self.act_layer = act_layer
        self.upsampling = upsampling
        self.flatten = nn.Flatten()
        
        layers = []
        layers.append(nn.Linear(1, ndense))
        layers.append(nn.ReLU())
        for _ in range(3):  # Repeat the following block 4 times
            layers.append(nn.Linear(ndense, ndense))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout1d(0.01))
        self.dense_stack = nn.Sequential(*layers)
        
        dense_large = []
        dense_large.append(nn.Linear(ndense, int(np.ceil(self.npix / self.upsampling)) * int(np.ceil(self.npix / self.upsampling)) * factor))
        dense_large.append(nn.ReLU())
        if dropout:
            dense_large.append(nn.Dropout1d(0.01))
        self.dense_large = nn.Sequential(*dense_large)
        
        self.reshape = nn.Unflatten(1, (factor, int(np.ceil(self.npix / self.upsampling)), int(np.ceil(self.npix / self.upsampling))))
            
        conv_layers = []
        conv_layers.append(nn.Conv2d(factor, nfilts, kernel_size=3, stride=1, padding='same'))
        if norm_type is not None:
            if norm_type == 'batch':
                conv_layers.append(nn.BatchNorm2d(nfilts))
            elif norm_type == 'layer':
                conv_layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))            
        conv_layers.append(nn.ReLU())        
        for _ in range(2):  # Repeat the following block 3 times
            conv_layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            if norm_type is not None:
                if norm_type == 'batch':
                    conv_layers.append(nn.BatchNorm2d(nfilts))
                elif norm_type == 'layer':
                    conv_layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))           
            conv_layers.append(nn.ReLU())        
        self.conv2d_stack_afterdense = nn.Sequential(*conv_layers)

        conv_layers = []
        for _ in range(3):  # Repeat the following block 3 times
            conv_layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            if norm_type is not None:
                if norm_type == 'batch':
                    conv_layers.append(nn.BatchNorm2d(nfilts))
                elif norm_type == 'layer':
                    conv_layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))          
            conv_layers.append(nn.ReLU())        
        self.conv2d_stack = nn.Sequential(*conv_layers)
        
        self.conv2d_final = nn.Conv2d(nfilts, nims, kernel_size=3, stride=1, padding='same')
        self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Sigmoid = nn.Sigmoid()
        
            
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_stack(x)
        x = self.dense_large(x)
        x = self.reshape(x)

        if self.upsampling == 4:
            x = self.upsample2D(x)
            x = self.conv2d_stack_afterdense(x)
            x = self.upsample2D(x)
            x = self.conv2d_stack(x)
        elif self.upsampling == 2: 
            x = self.upsample2D(x)
            x = self.conv2d_stack_afterdense(x)
        elif self.upsampling == 1: 
            x = self.conv2d_stack_afterdense(x)            
        x = self.conv2d_final(x)
        if self.act_layer == 'Sigmoid':
            x = self.Sigmoid(x)
        return(x)


class VolumeModel(nn.Module):
    
    """
    A simple learnable 3D volume model where each slice in the volume is trainable.
    Supports differential updates or direct use of the internal volume.

    Parameters
    ----------
    npix : int
        Number of pixels in each spatial dimension (assumes square slices).
    num_slices : int
        Number of slices along the depth axis.
    vol : torch.Tensor or None
        Optional initial volume of shape (num_slices, npix, npix). If None, initializes to zeros.
    device : str
        Device to place the model parameters on ('cuda' or 'cpu').

    Forward
    -------
    input_volume : torch.Tensor
        External volume to add to the internal volume (shape: num_slices, npix, npix).
    diff : bool
        If True, output is input_volume + self.volume. If False, output is self.volume only.

    Returns
    -------
    torch.Tensor
        Transformed volume of shape (num_slices, npix, npix).
    """
    
    def __init__(self, npix, num_slices, vol=None, device='cuda'):
        super(VolumeModel, self).__init__()
        self.num_slices = num_slices
        if vol is None:
            self.volume = nn.Parameter(torch.zeros((num_slices, npix, npix)).to(device))
        else:
            self.volume = nn.Parameter(vol)
    
    def forward(self, input_volume, diff = True):
        
        if diff:
            transformed_volume = input_volume + self.volume
        else:
            transformed_volume = self.volume
        
        return transformed_volume
    
    
class PeakModel(nn.Module):
    
    """
    A peak fitting model that represents parameterized 1D functions (e.g. Gaussian or Pseudo-Voigt)
    using learnable normalized parameters constrained to [0, 1]. Converts normalized parameters 
    to their physical range before evaluating the function.

    Parameters
    ----------
    prms : dict
        Dictionary containing:
        - 'val': torch.Tensor of shape (n_params, npix, npix), the initial normalized parameters.
        - 'min': dict of minimum values for each parameter.
        - 'max': dict of maximum values for each parameter.
    device : str
        Device to place parameters on ('cuda' or 'cpu').

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (N, 1) representing the x-axis values for function evaluation.
    model : str
        Peak function model to use: 'Gaussian' or 'PseudoVoigt'.

    Returns
    -------
    y : torch.Tensor
        Output tensor of shape (N, 1), the evaluated function for each pixel.
    """
    
    def __init__(self, prms, device='cuda'):

        super(PeakModel, self).__init__()  # Call the parent class's constructor first
        
        self.prms = nn.Parameter(prms['val'])
        self.min = prms['min']
        self.max = prms['max']
        self.nch = prms['val'].shape[0]
        self.npix = prms['val'].shape[1]
        
    def forward(self, x, model = 'Gaussian'):
    
        prms = torch.reshape(self.prms, (self.nch, self.npix*self.npix))
        prms = torch.transpose(prms, 1, 0)

        # Apply constraints to parameters
        prms = torch.clamp(prms, 0, 1)
 
        if model == 'Gaussian':
            y = ((self.min['Area'] + (self.max['Area']-self.min['Area'])*prms[:, 0:1]) * 
                 torch.exp(-(x - (self.min['Position'] + (self.max['Position'] - self.min['Position'])*prms[:, 1:2]))**2 
                                          / (2 * (self.min['FWHM'] + (self.max['FWHM'] - self.min['FWHM'])*prms[:, 2:3])**2)) + 
                 self.min['Slope'] + (self.max['Slope'] - self.min['Slope'])*prms[:, 3:4]*x + 
                 self.min['Intercept'] + (self.max['Intercept'] - self.min['Intercept'])*prms[:, 4:5])  

        elif model == 'PseudoVoigt':

            y = ((self.min['Fraction'] + (self.max['Fraction']-self.min['Fraction'])*prms[:, 5:6])*
                 (self.min['Area'] + (self.max['Area']-self.min['Area'])*prms[:, 0:1]) * 
                 torch.exp(-(x - (self.min['Position'] + (self.max['Position'] - self.min['Position'])*prms[:, 1:2]))**2 
                                          / (2 * (self.min['FWHM'] + (self.max['FWHM'] - self.min['FWHM'])*prms[:, 2:3])**2)) + 
                 (1-(self.min['Fraction'] + (self.max['Fraction']-self.min['Fraction'])*prms[:, 5:6]))*
                 ((self.min['Area'] + (self.max['Area']-self.min['Area'])*prms[:, 0:1])/ torch.pi) *
                 ( (self.min['FWHM'] + (self.max['FWHM'] - self.min['FWHM'])*prms[:, 2:3]) / 
                  (((x - (self.min['Position'] + (self.max['Position'] - self.min['Position'])*prms[:, 1:2]))**2) + 
                   (self.min['FWHM'] + (self.max['FWHM'] - self.min['FWHM'])*prms[:, 2:3])**2)) + 
                 self.min['Slope'] + (self.max['Slope'] - self.min['Slope'])*prms[:, 3:4]*x + 
                 self.min['Intercept'] + (self.max['Intercept'] - self.min['Intercept'])*prms[:, 4:5])
            
        return y
    
    
class ResNetBlock(nn.Module):
    
    """
    A basic 2D residual block consisting of two convolutional layers with optional normalization
    and ReLU activation, followed by a skip connection.

    Parameters
    ----------
    nfilts : int
        Number of filters in the convolutional layers.
    npix : int
        Spatial dimension (assumes square input for LayerNorm).
    kernel_size : int
        Size of convolutional kernel.
    stride : int
        Stride for the convolution.
    padding : int
        Padding to apply to convolution.
    norm_type : str or None
        Type of normalization: 'batch', 'layer', or None.

    Returns
    -------
    torch.Tensor
        Output tensor of same shape as input.
    """
        
    def __init__(self, nfilts, npix, kernel_size=3, stride=1, padding=1, norm_type=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(nfilts, nfilts, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(nfilts, nfilts, kernel_size, stride, padding)
        self.norm_type = norm_type
        self.npix = npix
        self.norm1 = self.add_norm_layer(nfilts)
        self.norm2 = self.add_norm_layer(nfilts)

    def add_norm_layer(self, nfilts):
        if self.norm_type == 'batch':
            return nn.BatchNorm2d(nfilts)
        elif self.norm_type == 'layer':
            return nn.LayerNorm([nfilts, self.npix, self.npix])
        return None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        out += residual  # Skip Connection
        out = F.relu(out)
        return out

class ResNet2D(nn.Module):
    
    
    """
    A 2D ResNet-style convolutional network for image processing tasks with configurable residual blocks
    and optional normalization.

    Parameters
    ----------
    npix : int
        Spatial size of the input (assumed square).
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    nfilts : int
        Number of filters in convolutional layers.
    n_res_blocks : int
        Number of ResNet blocks in the model.
    norm_type : str or None
        Type of normalization: 'batch', 'layer', or None.
    activation : str
        Final activation function: 'Linear' or 'Sigmoid'.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nch_out, npix, npix).
    """
        
    def __init__(self, npix, nch_in=1, nch_out=1, nfilts=32, n_res_blocks=4, norm_type='layer', activation='Linear'):
        super(ResNet2D, self).__init__()

        self.npix = npix
        self.n_res_blocks = n_res_blocks

        layers = [nn.Conv2d(nch_in, nfilts, kernel_size=3, stride=1, padding=1)]
        if norm_type is not None:
            layers.append(self.add_norm_layer(nfilts, norm_type))
        layers.append(nn.ReLU())

        for _ in range(n_res_blocks):
            layers.append(ResNetBlock(nfilts, npix=npix, norm_type=norm_type))

        layers.append(nn.Conv2d(nfilts, nch_out, kernel_size=3, stride=1, padding=1))
        if activation == 'Sigmoid':
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def add_norm_layer(self, nfilts, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm2d(nfilts)
        elif norm_type == 'layer':
            return nn.LayerNorm([nfilts, self.npix, self.npix])

    def forward(self, x, residual=False):
        out = self.model(x)
        if residual:
            out += x
        return out  
    

class ConvBlock(nn.Module):
    
    """
    A basic convolutional block with two Conv2D layers, LayerNorm (by default), and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spatial_dims : tuple[int, int]
        Spatial dimensions of the input (height, width).
    norm_type : str
        Normalization type: 'layer' (default) or 'batch'.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, out_channels, H, W).
    """
        
    def __init__(self, in_channels, out_channels, spatial_dims, norm_type='layer'):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm_type = norm_type
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.norm_layer1 = nn.LayerNorm([self.out_channels, self.spatial_dims[0], self.spatial_dims[1]])
        self.norm_layer2 = nn.LayerNorm([self.out_channels, self.spatial_dims[0], self.spatial_dims[1]])

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm_layer1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm_layer2(x)
        x = F.relu(x)
        return x

class DownBlock(nn.Module):
    
    """
    A downsampling block consisting of a Conv2D with stride=2 followed by a ConvBlock.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    spatial_dims : tuple[int, int]
        Spatial dimensions after downsampling.
    norm_type : str or None
        Type of normalization to use in the ConvBlock.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, out_channels, H/2, W/2).
    """
        
    def __init__(self, in_channels, out_channels, spatial_dims, norm_type=None):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_block = ConvBlock(out_channels, out_channels, spatial_dims, norm_type=norm_type)

    def forward(self, x):
        x = F.relu(self.down_conv(x))
        return self.conv_block(x)


class UpBlock(nn.Module):
    
    """
    An upsampling block with transposed convolution and concatenation with a skip connection (bridge),
    followed by a ConvBlock.

    Parameters
    ----------
    in_channels : int
        Number of channels to upsample.
    bridge_channels : int
        Number of channels in the bridge tensor (from encoder).
    out_channels : int
        Number of output channels after convolution.
    spatial_dims : tuple[int, int]
        Spatial dimensions of the output.
    norm_type : str
        Normalization type to use in ConvBlock: 'layer' or 'batch'.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, out_channels, H*2, W*2).
    """
        
    def __init__(self, in_channels, bridge_channels, out_channels, spatial_dims, norm_type='layer'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # The combined channels from the upsampled layer and the bridge
        combined_channels = out_channels + bridge_channels
        # Use the spatial dimensions for the ConvBlock
        self.conv_block = ConvBlock(combined_channels, out_channels, spatial_dims, norm_type=norm_type)
    def forward(self, x, bridge):
        x = self.up(x)
        # Determine padding for concatenation
        diffY = bridge.size()[2] - x.size()[2]
        diffX = bridge.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        out = torch.cat([bridge, x], dim=1)
        return self.conv_block(out)

    
class UNet2D(nn.Module):
    
    """
    A U-Net style 2D convolutional neural network with downsampling and upsampling paths and skip connections.
    Supports configurable depth and normalization.

    Parameters
    ----------
    nch_in : int
        Number of input channels.
    nch_out : int
        Number of output channels.
    npix : int
        Input spatial dimension (assumes square input).
    base_nfilts : int
        Number of filters in the base layer.
    num_blocks : int
        Number of downsampling and upsampling blocks.
    norm_type : str or None
        Normalization type: 'batch', 'layer', or None.
    activation : str
        Final activation function: 'Sigmoid' or 'Linear'.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape (batch_size, nch_in, npix, npix).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, nch_out, npix, npix).
    """
        
    def __init__(self, nch_in, nch_out, npix, base_nfilts=64, num_blocks=4, norm_type=None, activation='Linear'):
        super(UNet2D, self).__init__()

        spatial_dims = [npix, npix]  # Initial spatial dimensions
        self.initial_conv_block = ConvBlock(nch_in, base_nfilts, spatial_dims, norm_type=norm_type)
        self.down_blocks = nn.ModuleList()
        self.bridge_channels = []
        for i in range(num_blocks):
            spatial_dims = [s // 2 for s in spatial_dims]  # Halve spatial dimensions after down block
            self.down_blocks.append(DownBlock(base_nfilts, base_nfilts, spatial_dims, norm_type=norm_type))
            self.bridge_channels.append(base_nfilts)

        self.up_blocks = nn.ModuleList()
        # Adjust spatial_dims for UpBlocks
        for i in range(num_blocks-1, -1, -1):
            bridge_channels = self.bridge_channels[i]
            self.up_blocks.append(UpBlock(base_nfilts, bridge_channels, base_nfilts, spatial_dims, norm_type=norm_type))
            spatial_dims = [s * 2 for s in spatial_dims]  # Upsampling increases dimensions

        self.final_up_block = UpBlock(base_nfilts, bridge_channels,  base_nfilts, spatial_dims, norm_type=norm_type)

        # Final convolution layer
        self.final_conv = nn.Conv2d(base_nfilts, nch_out, kernel_size=1)
        self.activation = nn.Sigmoid() if activation == 'Sigmoid' else None

    def forward(self, x):
        bridges = []

        # Initial convolution
        initial_conv_output = self.initial_conv_block(x)

        # Downsampling
        x = initial_conv_output
        for down_block in self.down_blocks:
            x = down_block(x)
            bridges.append(x)

        bridges = bridges[::-1]

        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            bridge = bridges[i]
            x = up_block(x, bridge)

        # Upsampling for the initial convolution block
        x = self.final_up_block(x, initial_conv_output)

        # Final convolution layer
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    