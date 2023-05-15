# -*- coding: utf-8 -*-
"""
Losses for pytorch

@author: Antony Vamvakeros

"""
#%%

import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):

    '''
    Dice loss taken from https://github.com/usuyama/pytorch-unet/blob/master/loss.py
    '''
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
	
def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
	
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def ssim_loss(img1, img2, data_range=1.0, window_size=11, size_average=True):
    """
    Calculate the SSIM loss between two images using PyTorch.
    """
    k1 = 0.01
    k2 = 0.03
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    batch_size, num_channels, height, width = img1.size()
    kernel = torch.ones(num_channels, num_channels, window_size, window_size).to(img1.device)
    kernel /= kernel.sum()
    mu1 = F.conv2d(img1, kernel, padding='same', stride=1)
    mu2 = F.conv2d(img2, kernel, padding='same', stride=1)
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding='same', stride=1) - mu1 * mu1
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding='same', stride=1) - mu2 * mu2
    sigma12 = F.conv2d(img1 * img2, kernel, padding='same', stride=1) - mu1 * mu2
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return 1 - torch.mean(ssim_val)
    else:
        return 1 - ssim_val

