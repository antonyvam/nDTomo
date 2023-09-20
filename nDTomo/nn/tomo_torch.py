# -*- coding: utf-8 -*-
"""
Pytorch functions for tomography

Need to debug the angles for radon/iradon

@author: Antony Vamvakeros
"""

import torch
import torchvision.transforms.functional as TF
from numpy import vstack
import torch.nn.functional as F

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
    Create the torch sparse A matrix and its transpose
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

    return(Atorch)

def Sino_torch(Atorch, im, ntr, npr):

    '''
    Prepare the image for cuda and create the sinogram using the A matrix
    '''

    imt = torch.from_numpy(im).float()
    imt = torch.reshape(imt, (ntr*ntr , 1))
    s = Amatrix_sino(Atorch, imt.cuda(), npr, ntr)

    return(s)

def Amatrix_sino(Atorch, im, npr, ntr):

    '''
    Create sinogram using the A matrix
    '''

    stf = torch.matmul(Atorch, im)
    stf = torch.reshape(stf, (npr, ntr))

    return(stf)

def Amatrix_rec(AtorchT, s, ntr):

    '''
    Create reconstructed image using the A matrix
    '''

    rec = torch.matmul(AtorchT,s)
    rec = torch.reshape(rec, (ntr, ntr))

    return(rec)

def RotMat(theta):

    '''
    Create 2D rotation matrix
    '''

    theta = torch.tensor(theta)
    rotmat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])
    return(rotmat)


def imrotate_torch(im, theta, dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor):

    '''
    Rotate 2D image using the rotation matrix
    '''

    rot_mat = RotMat(theta)[None, ...].type(dtype).repeat(im.shape[0],1,1)
    grid = F.affine_grid(rot_mat, im.size()).type(dtype)
    imr = F.grid_sample(im, grid)
    return(imr)