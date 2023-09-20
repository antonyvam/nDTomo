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
    
    def __init__(self, npix, nch_in=1, nch_out=1, nfilts=32, nlayers=4, norm_type='layer', activation='Linear'):
        
        super(CNN2D, self).__init__()
        
        self.npix = npix
        
        layers = []
        layers.append(nn.Conv2d(nch_in, nfilts, kernel_size=3, stride=1, padding=1))  # 'same' padding in PyTorch is usually done by manually specifying the padding
        if norm_type is not None:
            self.add_norm_layer(layers, nfilts, norm_type)
        layers.append(nn.ReLU())
        
        for layer in range(nlayers):
            layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding=1))
            if norm_type is not None:
                self.add_norm_layer(layers, nfilts, norm_type)
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(nfilts, nch_out, kernel_size=3, stride=1, padding=1))
        
        if activation == 'Sigmoid':
            layers.append(nn.Sigmoid())
        
        self.cnn2d = nn.Sequential(*layers)

    def add_norm_layer(self, layers, nfilts, norm_type):
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(nfilts))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))
        else:
            raise ValueError('Invalid normalization type')
            
    def forward(self, x, residual=False):
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
    
    def __init__(self, npix, factor=8, nims=5, nfilts = 32, ndense = 64, dropout=True, norm_type='layer', upsampling = 4):
        
        super(SD2I, self).__init__()
        
        self.upsampling = upsampling
        self.flatten = nn.Flatten()
        
        layers = []
        for _ in range(4):  # Repeat the following block 4 times
            layers.append(nn.Linear(1, ndense))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout1d(0.01))

        self.dense_stack = nn.Sequential(*layers)
        dense_large = []
        dense_large.append(nn.Linear(ndense, int(np.ceil(npix / self.upsampling)) * int(np.ceil(npix / self.upsampling)) * factor))
        dense_large.append(nn.ReLU())
        if dropout:
            dense_large.append(nn.Dropout1d(0.01))
        self.dense_large = dense_large
        self.reshape = nn.Unflatten(1, (factor, int(np.ceil(npix / self.upsampling)), int(np.ceil(npix / self.upsampling))))
            
        conv_layers = []
        conv_layers.append(nn.Conv2d(factor, nfilts, kernel_size=3, stride=1, padding='same'))
        if norm_type is not None:
            self.add_norm_layer(layers, nfilts, norm_type)            
        conv_layers.append(nn.ReLU())        
        for _ in range(2):  # Repeat the following block 3 times
            conv_layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            if norm_type is not None:
                self.add_norm_layer(layers, nfilts, norm_type)            
            conv_layers.append(nn.ReLU())        
        self.conv2d_stack_afterdense = nn.Sequential(*conv_layers)

        conv_layers = []
        for _ in range(3):  # Repeat the following block 3 times
            conv_layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            if norm_type is not None:
                self.add_norm_layer(layers, nfilts, norm_type)            
            conv_layers.append(nn.ReLU())        
        self.conv2d_stack = nn.Sequential(*conv_layers)
        
        self.conv2d_final = nn.Conv2d(nfilts, nims, kernel_size=3, stride=1, padding='same')
        self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Sigmoid = nn.Sigmoid()
        
    def add_norm_layer(self, layers, nfilts, norm_type):
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(nfilts))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([nfilts, self.npix, self.npix]))
        else:
            raise ValueError('Invalid normalization type')
            
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
    
    
    
    
    
    
    
    
    
    
    
    