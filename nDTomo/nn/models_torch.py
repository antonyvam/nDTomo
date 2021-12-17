# -*- coding: utf-8 -*-
"""
Neural networks models

@author: Antony Vamvakeros
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%

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
        x7 = self.conv2d_dual(x7)

        x8 = self.up(x7)
        x8 = self.crop(x8, x2)
        x8 = self.conv2d_dual(x8)
        
        x9 = self.up(x8)
        x9 = self.crop(x9, x1)
        x9 = self.conv2d_dual(x9)
        
        x = self.out(x9)

        return(x)
