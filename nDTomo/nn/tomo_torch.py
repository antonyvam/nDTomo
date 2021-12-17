# -*- coding: utf-8 -*-
"""
Pytorch functions for tomography

Need to debug the angles for radon/iradon
"""

import torch
import torchvision.transforms.functional as TF
from numpy import vstack

def radon_torch(img, theta, nproj):

    imgn = torch.zeros(img.shape[1], img.shape[2], nproj).cuda()

    for ii in range(nproj):
        
        imgn[:,:,ii] = TF.rotate(img, theta[ii].item())

    imgn = torch.sum(imgn, 1)

    return(imgn)

def radon_torch(s, theta, nproj):

    sn = torch.reshape(s, (1, 1, s.shape[0], s.shape[1]))
    sn = torch.tile(sn, (1, sn.shape[2], 1, 1))

    bp = torch.zeros(sn.shape[1], sn.shape[1]).cuda()

    for ii in range(nproj):

        bp = bp + TF.rotate(sn[:,:,:,ii], theta[ii].item())

    bp = bp / nproj

    return(bp)

def Amatrix_torch(A, gpu = True):

    '''
    Create the torch sparse A matrix and its tranpose
    This can be improved
    '''

    Acoo = A.tocoo()

    values = Acoo.data
    indices = vstack((Acoo.row, Acoo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = Acoo.shape

    if gpu == True:

        Atorch = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()

    else:

        Atorch = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    AtorchT = torch.transpose(Atorch, 1, 0)

    return(Atorch, AtorchT)


def Amatrix_sino(Atorch, im, npr, ntr):

    stf = torch.matmul(Atorch, im)
    stf = torch.reshape(stf, (npr, ntr))

    return(stf)

def Amatrix_rec(AtorchT, s, ntr):

    rec = torch.matmul(AtorchT,s)
    rec = torch.reshape(rec, (ntr, ntr))

    return(rec)
