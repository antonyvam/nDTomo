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


def back_project_2D(sinogram, angles, output_size, normalize=False, device='cuda'):
    """
    Back projector in PyTorch for 2D sinogram using rotate and stack.

    Parameters
    ----------
    sinogram : torch.Tensor
        Tensor of shape (H, len(angles)) on GPU.
    angles : array-like
        List of angles in degrees.
    output_size : int
        Desired reconstructed image width and height.
    normalize : bool
        Whether to apply normalization by pi / n_angles.
    device : str
        CUDA device.

    Returns
    -------
    recon : torch.Tensor
        Tensor of shape (1, 1, output_size, output_size).
    """
    n_detectors, n_angles = sinogram.shape
    recon = torch.zeros((1, 1, output_size, output_size), device=device)

    for i, theta in enumerate(angles):
        proj = sinogram[:, i]
        proj_2d = proj.view(n_detectors, 1).repeat(1, output_size).unsqueeze(0).unsqueeze(0)
        rotated = rotate(proj_2d, theta + 90, interpolation=InterpolationMode.BILINEAR)
        recon += rotated

    if normalize:
        recon *= torch.pi / len(angles)

    return recon



def forwardproject_torch(vol, angles, Amat=None, grid_scaled=None, device='cuda'):
    """
    Forward projection (Radon transform) in PyTorch for 2D or 3D input volumes.

    This function supports both:
    - 2D inputs: performs projection via image rotation and summation.
    - 3D volumes: assumes shape (z, x, y) and performs slice-wise projection for each angle.

    Parameters
    ----------
    vol : torch.Tensor
        Input tensor representing the volume:
        - Shape (H, W) for 2D projection.
        - Shape (Z, H, W) for 3D projection (multiple slices).
        Assumes square images for simplicity in rotation logic.
    angles : array-like
        Sequence of projection angles in degrees.
    Amat : torch.sparse.Tensor or None, optional
        Optional sparse matrix for projection in 2D case. If provided, it overrides the geometric simulation.
    grid_scaled : torch.Tensor or None, optional
        If provided (for 3D), uses this grid for sampling via F.grid_sample.
        Should match shape and spacing of input for distortion correction or alignment.
    device : str, optional
        Device to run the computation on. Default is 'cuda'.

    Returns
    -------
    s : torch.Tensor
        Forward projected sinogram:
        - Shape (npix, n_angles) for 2D.
        - Shape (nbins, npix, n_angles) for 3D.
    """
    vol = vol.to(device)
    dims = vol.shape
    angles = [float(a) for a in angles]  # ensure float angles

    if len(dims) == 3:
        # 3D case: (Z, X, Y)
        nbins, npix, _ = dims
        s = torch.zeros((nbins, npix, len(angles)), device=device)

        for i, theta in enumerate(angles):
            vol_rot = rotate(vol, theta, interpolation=InterpolationMode.BILINEAR)
            vol_rot = vol_rot.view(1, 1, nbins, npix, npix)

            if grid_scaled is not None:
                voli = F.grid_sample(vol_rot, grid_scaled, mode='bilinear', align_corners=True)
                s[:, :, i] = torch.sum(voli, dim=4)[0, 0]
            else:
                s[:, :, i] = torch.sum(vol_rot, dim=4)[0, 0]

    elif len(dims) == 2:
        # 2D case: (X, Y)
        npix = dims[0]

        if Amat is not None:
            s = Amat @ vol.view(-1, 1)  # flattened projection
        else:
            s = torch.zeros((npix, len(angles)), device=device)
            for i, theta in enumerate(angles):
                vol_rot = rotate(vol, theta, interpolation=InterpolationMode.BILINEAR)
                vol_rot = vol_rot.view(1, 1, npix, npix)
                s[:, i] = torch.sum(vol_rot, dim=3)[0, 0]

    else:
        raise ValueError("Input volume must be either 2D (H, W) or 3D (Z, H, W)")

    return s

def sirt_pytorch(sinogram, angles, output_size, n_iter=20, relax=1.0, epsilon=1e-6, device='cuda'):
    W_voxel = back_project_2D(forward_project_2D(
        torch.ones((1, 1, output_size, output_size), device=device), angles, device),
        angles, output_size, normalize=False, device=device)

    x = torch.zeros((1, 1, output_size, output_size), device=device)
    for _ in range(n_iter):
        r = sinogram - forward_project_2D(x, angles, device)
        correction = back_project_2D(r, angles, output_size, normalize=False, device=device)
        x += relax * correction / (W_voxel + epsilon)

    return x

def cgls_pytorch(sinogram, angles, output_size, n_iter=10, device='cuda'):
    def forward(x): return forward_project_2D(x, angles, device)
    def backward(y): return back_project_2D(y, angles, output_size, normalize=False, device=device)

    x = torch.zeros((1, 1, output_size, output_size), device=device)
    b = sinogram
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
