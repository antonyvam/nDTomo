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