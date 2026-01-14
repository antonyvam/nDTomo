# -*- coding: utf-8 -*-
"""
PyTorch functions for 2D and 3D tomography and Radon transform simulation.

This module includes differentiable and non-differentiable implementations of:
- Forward and back projection routines (Radon and inverse Radon transforms) in 2D and 3D.
- Iterative reconstruction methods such as SIRT and CGLS using functional forward/back projectors.
- Sparse matrix-based forward and backward operations (A-matrix formulation) using PyTorch sparse tensors.
- Utility functions for constructing sparse system matrices, rotating images, and defining affine transforms.

Main features:
- Differentiable 3D forward and backward projectors using `torchvision.transforms.functional.rotate`.
- Support for iterative solvers: SIRT (with normalization) and CGLS.
- Conversion utilities for using SciPy sparse matrices in PyTorch (e.g., `Amatrix_torch`, `Sino_torch`, `Amatrix_rec`).
- Grid-based rotation of images via affine transformation (`imrotate_torch`).
- Compatible with both CPU and CUDA devices.

Author: Antony Vamvakeros
"""

import torch
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from numpy import vstack
import numpy as np

def forward_project_3D(vol, angles, npix, nch, device='cuda'):

    """
    Perform forward projection (Radon transform) of a 3D volume using PyTorch.

    Parameters
    ----------
    vol : torch.Tensor
        Input volume of shape (1, nch, npix, npix), where:
        - 1 is the batch dimension,
        - nch is the number of channels or slices (e.g., spectral bins or time steps),
        - npix x npix is the spatial resolution of each slice.
    angles : list or ndarray
        List of projection angles in degrees.
    npix : int
        Number of pixels along each projection axis (image size).
    nch : int
        Number of slices or channels in the volume.
    device : str, optional
        PyTorch device string (default: 'cuda').

    Returns
    -------
    sinos : torch.Tensor
        Simulated sinogram of shape (nch, npix, len(angles)), where each slice
        corresponds to a different channel/slice and each column is a projection.
    """

    sinos = torch.zeros((nch, npix, len(angles)), device=device)
    for angle in range(len(angles)):
        vol_rot = rotate(vol, float(angles[angle]), interpolation=InterpolationMode.BILINEAR)
        sinos[:,:,angle] = torch.sum(vol_rot, dim=3)[0,:,:]
    return sinos

def back_project_3D(sinos, angles, npix, nch, device='cuda'):

    """
    Perform backprojection (inverse Radon transform) of a 3D sinogram using PyTorch.

    Parameters
    ----------
    sinos : torch.Tensor
        Input sinogram of shape (nch, npix, len(angles)), where:
        - nch is the number of slices or channels,
        - npix is the number of detector elements,
        - len(angles) is the number of projection angles.
    angles : list or ndarray
        List of projection angles in degrees.
    npix : int
        Number of pixels in the output reconstructed image.
    nch : int
        Number of slices or channels in the volume.
    device : str, optional
        PyTorch device string (default: 'cuda').

    Returns
    -------
    vol : torch.Tensor
        Reconstructed 3D volume of shape (nch, npix, npix). Each channel corresponds to
        a separate slice, reconstructed via filtered or unfiltered backprojection.
    """
    vol = torch.zeros((1, nch, npix, npix), device=device)
    for angle in range(len(angles)):
        vol_rot = sinos[:,:,angle].unsqueeze(0).unsqueeze(0)
        vol_rot = torch.transpose(vol_rot, 2, 1)
        vol_rot = vol_rot.repeat(1, 1, npix, 1)
        vol_rot = torch.transpose(vol_rot, 3, 2)
        vol_rot = rotate(vol_rot, 
                         -float(angles[angle]), interpolation=InterpolationMode.BILINEAR)
        vol += vol_rot
    return vol[0, :, :, :]


# Compute W_ray for SIRT
def compute_W_ray(angles, npix, nch, device='cuda'):
    """
    Compute ray normalization weights for each voxel via forward projection of a constant volume.

    This function simulates the accumulation of contributions each detector sees from a
    uniform volume, useful for SIRT or SART-type normalization.

    Parameters
    ----------
    angles : list or ndarray
        List of projection angles in degrees.
    npix : int
        Number of pixels in each dimension of the image (image size).
    nch : int
        Number of slices or channels in the 3D volume.
    device : str, optional
        PyTorch device string (default: 'cuda').

    Returns
    -------
    W_ray : torch.Tensor
        Weighting map of shape (nch, npix, len(angles)) representing forward projection of ones.
    """    
    vol = torch.ones((1, nch, npix, npix), dtype=torch.float32, device=device)
    W_ray = forward_project_3D(vol, angles, npix, nch, device)
    return W_ray


def sirt_pytorch_functional(sinos, angles, npix, nch=1, n_iter=20, relax=0.01, epsilon=1e-6, device='cuda'):
    """
    SIRT reconstruction using PyTorch with function-based forward and backward projectors.

    Parameters
    ----------
    sinos : torch.Tensor
        Input sinogram tensor of shape (nch, npix, n_angles), e.g. (1, 151, 180).
    angles : list or ndarray
        List of projection angles in degrees.
    npix : int
        Width/height of the reconstructed image.
    nch : int
        Number of slices or channels in the volume (default = 1).
    n_iter : int
        Number of SIRT iterations.
    relax : float
        Relaxation factor (typically small, e.g., 0.01).
    epsilon : float
        Small number to avoid division by zero.
    device : str
        Computation device, e.g. 'cuda'.

    Returns
    -------
    torch.Tensor
        Reconstructed volume of shape (nch, npix, npix).
    """
    sinos = sinos.to(device)
    W_ray = compute_W_ray(angles, npix, nch, device=device)

    vol = torch.zeros((1, nch, npix, npix), dtype=torch.float32, device=device)

    for _ in range(n_iter):
        sim = forward_project_3D(vol, angles, npix, nch, device)
        residual = sinos - sim
        correction = back_project_3D(residual / (W_ray + epsilon), angles, npix, nch, device)
        vol += relax * correction

    if device == 'cuda':
        vol = vol.cpu() 

    vol = vol.squeeze().numpy()
    vol = np.transpose(vol)

    return vol  # shape: (nch, npix, npix)


def cgls_pytorch_functional(sinos, angles, npix, nch=1, n_iter=10, device='cuda'):
    """
    CGLS reconstruction using PyTorch with functional forward and back projectors.

    Parameters
    ----------
    sinos : torch.Tensor
        Input sinogram tensor of shape (nch, npix, n_angles), e.g., (1, 151, 180).
    angles : list or ndarray
        List of projection angles in degrees.
    npix : int
        Width/height of the reconstructed image.
    nch : int
        Number of slices or channels in the volume (default = 1).
    n_iter : int
        Number of CGLS iterations.
    device : str
        Computation device, e.g., 'cuda'.

    Returns
    -------
    torch.Tensor
        Reconstructed volume of shape (nch, npix, npix).
    """
    sinos = sinos.to(device)

    def forward(x):
        x_batched = x.unsqueeze(0)  # (1, nch, npix, npix)
        return forward_project_3D(x_batched, angles, npix, nch, device)

    def backward(y):
        return back_project_3D(y, angles, npix, nch, device)

    x = torch.zeros((nch, npix, npix), dtype=torch.float32, device=device)

    b = sinos
    r = b - forward(x)
    p = backward(r)
    d = p.clone()
    delta_new = torch.sum(d * d)

    for _ in range(n_iter):
        q = forward(d)
        alpha = delta_new / torch.sum(q * q)
        x += alpha * d
        r -= alpha * q
        s = backward(r)
        delta_old = delta_new
        delta_new = torch.sum(s * s)
        beta = delta_new / delta_old
        d = s + beta * d

    if device == 'cuda':
        x = x.cpu() 

    x = x.squeeze().numpy()
    x = np.transpose(x)

    return x

def Amatrix_torch(A, gpu=True):
    """
    Converts a SciPy sparse matrix A to a PyTorch sparse tensor.

    Parameters
    ----------
    A : scipy.sparse matrix
        Input sparse matrix in COO format (or convertible to COO).
    gpu : bool, optional
        If True, moves the tensor to CUDA. Default is True.

    Returns
    -------
    Atorch : torch.sparse.FloatTensor
        PyTorch sparse tensor version of A.
    """
    Acoo = A.tocoo()
    values = Acoo.data
    indices = vstack((Acoo.row, Acoo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = Acoo.shape

    Atorch = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return Atorch.cuda() if gpu else Atorch

def Sino_torch(Atorch, im, ntr, npr):
    """
    Generates a sinogram by applying a torch sparse A matrix to an image.

    Parameters
    ----------
    Atorch : torch.sparse.FloatTensor
        Sparse A matrix in torch format.
    im : ndarray
        Input image as a NumPy array of shape (ntr, ntr).
    ntr : int
        Number of translation steps (image side length).
    npr : int
        Number of projections.

    Returns
    -------
    s : torch.Tensor
        Sinogram as a tensor of shape (npr, ntr).
    """
    imt = torch.from_numpy(im).float().reshape(ntr * ntr, 1).cuda()
    s = Amatrix_sino(Atorch, imt, npr, ntr)
    return s

def Amatrix_sino(Atorch, im, npr, ntr):
    """
    Computes the sinogram using matrix multiplication with a sparse A matrix.

    Parameters
    ----------
    Atorch : torch.sparse.FloatTensor
        Sparse system matrix A (shape: [npr*ntr, ntr*ntr]).
    im : torch.Tensor
        Flattened image tensor of shape (ntr*ntr, 1).
    npr : int
        Number of projections (angles).
    ntr : int
        Number of translation steps.

    Returns
    -------
    stf : torch.Tensor
        Sinogram tensor of shape (npr, ntr).
    """
    stf = torch.matmul(Atorch, im)
    stf = stf.reshape(npr, ntr)
    return stf

def Amatrix_rec(AtorchT, s, ntr):
    """
    Reconstructs an image from a sinogram using the transpose of the A matrix.

    Parameters
    ----------
    AtorchT : torch.sparse.FloatTensor
        Transpose of the system matrix A (shape: [ntr*ntr, npr*ntr]).
    s : torch.Tensor
        Sinogram of shape (npr, ntr).
    ntr : int
        Number of translation steps (output image side length).

    Returns
    -------
    rec : torch.Tensor
        Reconstructed image tensor of shape (ntr, ntr).
    """
    rec = torch.matmul(AtorchT, s.view(-1, 1))
    return rec.reshape(ntr, ntr)

def RotMat(theta):
    """
    Creates a 2D rotation matrix for use in affine transformations.

    Parameters
    ----------
    theta : float or torch.Tensor
        Rotation angle in radians.

    Returns
    -------
    rotmat : torch.Tensor
        A 2×3 affine rotation matrix tensor.
    """
    theta = torch.tensor(theta)
    rotmat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                           [torch.sin(theta),  torch.cos(theta), 0]])
    return rotmat


def imrotate_torch(im, theta, dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor):
    """
    Rotates a 2D image (or batch) using an affine transformation.

    Parameters
    ----------
    im : torch.Tensor
        Input tensor of shape (N, C, H, W).
    theta : float
        Rotation angle in radians.
    dtype : torch dtype, optional
        Data type of the affine matrix and grid. Default is float tensor on GPU.

    Returns
    -------
    imr : torch.Tensor
        Rotated image of the same shape as input.
    """
    rot_mat = RotMat(theta)[None, ...].type(dtype).repeat(im.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, im.size(), align_corners=True).type(dtype)
    imr = F.grid_sample(im, grid, align_corners=True)
    return imr



def create_torch_Amat(Acoo, values, indices, shape, device='cuda'):
    """
    Constructs a PyTorch sparse A matrix from COO components.

    Parameters
    ----------
    Acoo : scipy.sparse.coo_matrix
        Sparse matrix object (for context).
    values : ndarray
        Non-zero values of the matrix.
    indices : ndarray
        2×N array of row and column indices.
    shape : tuple
        Shape of the matrix.
    device : str, optional
        Target device ('cuda' or 'cpu').

    Returns
    -------
    Amat : torch.sparse.FloatTensor
        Constructed sparse matrix on the target device.
    """
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
