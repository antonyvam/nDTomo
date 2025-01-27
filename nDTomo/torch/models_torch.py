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


class UNet(nn.Module):

    def __init__(self):
        
        super(UNet, self).__init__()
        
        self.conv2d_initial = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.conv2d_down = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                )
            
        self.conv2d_dual = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.out = nn.Conv2d(64, 1, kernel_size=1)
    
    def crop(self, x1, x2):

        '''
        Taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        '''

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        xn = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return(xn)
        
    def forward(self, x):
        
        x1 = self.conv2d_initial(x)
        x2 = self.conv2d_down(x1)
        x3 = self.conv2d_down(x2)
        x4 = self.conv2d_down(x3)
        x5 = self.conv2d_down(x4)

        x6 = self.up(x5)
        x6 = self.crop(x6, x4)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.conv2d_dual(x6)

        x7 = self.up(x6)
        x7 = self.crop(x7, x3)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.conv2d_dual(x7)

        x8 = self.up(x7)
        x8 = self.crop(x8, x2)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.conv2d_dual(x8)
        
        x9 = self.up(x8)
        x9 = self.crop(x9, x1)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.conv2d_dual(x9)
        
        x = self.out(x9)

        return(x)



class Autoencoder(nn.Module):

    def __init__(self):
        
        super(Autoencoder, self).__init__()
        
        self.conv2d_initial = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.conv2d_down = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                )
            
        self.conv2d_dual = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )

        self.out = nn.Conv2d(64, 1, kernel_size=1)
    
    def crop(self, x1, x2):

        '''
        Taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        '''

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        xn = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return(xn)
        
    def forward(self, x):
        
        x1 = self.conv2d_initial(x)
        x2 = self.conv2d_down(x1)
        x3 = self.conv2d_down(x2)
        x4 = self.conv2d_down(x3)
        x5 = self.conv2d_down(x4)

        x6 = self.up(x5)
        x6 = self.crop(x6, x4)
        x6 = self.conv2d_dual(x6)

        x7 = self.up(x6)
        x7 = self.crop(x7, x3)
        x7 = self.conv2d_dual(x7)

        x8 = self.up(x7)
        x8 = self.crop(x8, x2)
        x8 = self.conv2d_dual(x8)
        
        x9 = self.up(x8)
        x9 = self.crop(x9, x1)
        x9 = self.conv2d_dual(x9)
        
        x = self.out(x9)

        return(x)


class CNN1D(nn.Module):
    
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
    
    '''
    Requires npix and num_slices
    '''
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
    
    '''
    Requires npix and num_slices
    '''
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
    def __init__(self, in_channels, out_channels, spatial_dims, norm_type=None):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_block = ConvBlock(out_channels, out_channels, spatial_dims, norm_type=norm_type)

    def forward(self, x):
        x = F.relu(self.down_conv(x))
        return self.conv_block(x)


class UpBlock(nn.Module):
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
    