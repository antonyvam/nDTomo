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


class CNN2D(nn.Module):
    
    def __init__(self, npix, nch=5, nfilts=32, nlayers =4):
        
        super(CNN2D, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(nch, nfilts, kernel_size=3, stride=1, padding='same'))
        layers.append(nn.BatchNorm2d(nfilts))
        layers.append(nn.ReLU())
        
        for layer in range(nlayers):
            layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            layers.append(nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'))
            layers.append(nn.BatchNorm2d(nfilts))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(nfilts, nch, kernel_size=3, stride=1, padding='same'))
        layers.append(nn.Sigmoid())
        
        self.cnn2d = nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.cnn2d(x)
        
        return(out)


class SD2I(nn.Module):
    def __init__(self, npix, factor=8, nims=1, nfilts = 128, ndense = 128):
        
        super(SD2I, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(1, ndense),
            nn.ReLU(),
            nn.Linear(ndense, ndense),
            nn.ReLU(),
            nn.Linear(ndense, ndense),
            nn.ReLU(),
            nn.Linear(ndense, ndense),
            nn.ReLU(),            
        )
        self.dense_large = nn.Sequential(nn.Linear(ndense, int(np.ceil(npix / 4)) * int(np.ceil(npix / 4)) * factor),
                                         nn.ReLU()
                                         )

        self.reshape = nn.Unflatten(1, (factor, int(np.ceil(npix / 4)), int(np.ceil(npix / 4))))
        self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.conv2d_stack_afterdense = nn.Sequential(
            nn.Conv2d(factor, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
            )
        self.conv2d_stack = nn.Sequential(
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
            )
        self.conv2d_final = nn.Conv2d(nfilts, nims, kernel_size=3, stride=1, padding='same')
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_stack(x)
        x = self.dense_large(x)
        x = self.reshape(x)
        x = self.upsample2D(x)
        x = self.conv2d_stack_afterdense(x)
        x = self.upsample2D(x)
        x = self.conv2d_stack(x)
        x = self.conv2d_final(x)
        return(x)
    
    

class SD2I_peaks(nn.Module):
    def __init__(self, npix, factor=8, nims=5, nfilts = 128, ndense = 128):
        
        super(SD2I_peaks, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_stack = nn.Sequential(
            nn.Linear(1, ndense),
            nn.ReLU(),
            nn.Dropout1d(0.01),
            nn.Linear(ndense, ndense),
            nn.ReLU(),
            nn.Dropout1d(0.01),
            nn.Linear(ndense, ndense),
            nn.ReLU(),
            nn.Dropout1d(0.01),
            nn.Linear(ndense, ndense),
            nn.ReLU(),            
            nn.Dropout1d(0.01),
        )
        self.dense_large = nn.Sequential(nn.Linear(ndense, int(np.ceil(npix / 4)) * int(np.ceil(npix / 4)) * factor),
                                         nn.ReLU(),
                                         nn.Dropout1d(0.01)
                                         )

        self.reshape = nn.Unflatten(1, (factor, int(np.ceil(npix / 4)), int(np.ceil(npix / 4))))
        self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.conv2d_stack_afterdense = nn.Sequential(
            nn.Conv2d(factor, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU()
            )
        self.conv2d_stack = nn.Sequential(
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU(),
            nn.Conv2d(nfilts, nfilts, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(nfilts),
            nn.ReLU()
            )
        self.conv2d_final = nn.Conv2d(nfilts, nims, kernel_size=3, stride=1, padding='same')
        self.Sigmoid = nn.Sigmoid()
        self.BatchNorm2d = nn.BatchNorm2d(nfilts)
        self.BatchNorm1d = nn.BatchNorm1d(nfilts)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense_stack(x)
        x = self.dense_large(x)
        x = self.reshape(x)
        x = self.upsample2D(x)
        x = self.conv2d_stack_afterdense(x)
        x = self.upsample2D(x)
        x = self.conv2d_stack(x)
        x = self.conv2d_final(x)
        x = self.Sigmoid(x)
        return(x)
    