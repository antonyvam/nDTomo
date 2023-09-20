# -*- coding: utf-8 -*-
"""
Pytorch functions for tomography

Need to debug the angles for radon/iradon

@author: Antony Vamvakeros
"""

import torch
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from numpy import vstack
import numpy as np

def iradon(s, theta, nproj):

    sn = torch.reshape(s, (1, 1, s.shape[0], s.shape[1]))
    sn = torch.tile(sn, (1, sn.shape[2], 1, 1))

    bp = torch.zeros(sn.shape[1], sn.shape[1]).cuda()

    for ii in range(nproj):

        bp = bp + rotate(sn[:,:,:,ii], theta[ii].item())

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


def radon(vol, angles, Amat = None, grid_scaled=None, device='cuda'):

    '''
    If vol has 3 dimensions, then it should be in the form of (z, x, y)
    '''

    dims = vol.shape    

    if len(dims) == 3:
        
        nbins = vol.shape[0]
        npix = vol.shape[1]
        
        s = torch.zeros((nbins, npix, len(angles)), device=device)
        
        for angle in range(len(angles)):
             
            vol_rot = rotate(vol, float(angles[angle]), interpolation=InterpolationMode.BILINEAR)
            vol_rot = torch.reshape(vol_rot, (1, 1, vol.shape[0], vol.shape[1], vol.shape[1]))
            
            if grid_scaled is not None:
                voli = F.grid_sample(vol_rot, grid_scaled, mode='bilinear')    
                s[:,:,angle] = torch.sum(voli, dim=4)[0,0,:,:]
            else:
                s[:,:,angle] = torch.sum(vol_rot, dim=4)[0,0,:,:]

    elif len(dims) == 2:

        npix = vol.shape[0]

        s = torch.zeros((npix, len(angles)), device=device)
        
        for angle in range(len(angles)):
             
            vol_rot = rotate(vol, float(angles[angle]), interpolation=InterpolationMode.BILINEAR)
            vol_rot = torch.reshape(vol_rot, (1, 1, vol.shape[0], vol.shape[0]))
            
            s[:,angle] = torch.sum(vol_rot, dim=3)[0,0,:,:]

        
    return(s)



def grid(sinos, pgrid, Z, Z_start, device='cuda'):

    H = sinos.shape[0]
    W = sinos.shape[0]
    
    npix = sinos.shape[0]
    pgrid = np.float32(pgrid)
    pgrid = pgrid.transpose([2,0,1])
    pgrid = np.flip(pgrid, 1)
        
    tth = pgrid[:,int(npix/2),int(npix/2)]
    
    pgrid = pgrid[Z_start : Z, 0:H, 0:W]
    pgrid = torch.from_numpy(pgrid.copy())
    sinos = np.float32(sinos)
    sinos = sinos[:, :, Z_start : Z].transpose([2,0,1])
    
    sinos =  torch.from_numpy(sinos)
    
    tth = tth[Z_start : Z]
    npix = sinos.shape[1]
            
    Hv = torch.arange(0, W, device=device, dtype=torch.float32)
    Wv = torch.arange(0, H, device=device, dtype=torch.float32)
    Zv = torch.tensor(tth, dtype=torch.float32 ,device=device)
    
    grid_z, grid_y, grid_x = torch.meshgrid(Zv, Hv, Wv)
    grid = torch.stack((pgrid.to(device), grid_y, grid_x), 3).float()  # W(x), H(y), 2
    grid.requires_grad = False
    
    grid_x = 2.0 * grid[:, :, :, 2] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0
    max_value = torch.max(grid_z) - torch.min(grid_z)
    grid_z = grid[:, :, :, 0] - torch.min(grid_z)
    grid_z = 2.0 * grid_z/ max_value - 1.0
    grid_z[grid_z<-1] = -1
    grid_z[grid_z>1] = 1
    
    grid_scaled = torch.stack((grid_x, grid_y, grid_z), dim=3)
    grid_scaled = torch.reshape(grid_scaled, (1, grid_scaled.shape[0], grid_scaled.shape[1], grid_scaled.shape[2], grid_scaled.shape[3]))

    return(grid_scaled)



