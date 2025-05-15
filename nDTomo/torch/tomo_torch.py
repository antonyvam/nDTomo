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
        # 2D case: (H, W)
        npix = dims[0]

        if Amat is not None:
            s = Amat @ vol.view(-1, 1)
        else:
            s = torch.zeros((npix, len(angles)), device=device)
            for i, theta in enumerate(angles):
                vol_in = vol.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
                vol_rot = rotate(vol_in, theta, interpolation=InterpolationMode.BILINEAR)
                s[:, i] = torch.sum(vol_rot, dim=3)[0, 0]
    else:
        raise ValueError("Input volume must be either 2D (H, W) or 3D (Z, H, W)")

    return s


def backproject_torch(sinogram, angles, output_size, grid_scaled=None, normalize=False, device='cuda'):
    """
    Backprojection (inverse Radon transform) in PyTorch for 2D or 3D sinograms.

    This function supports both:
    - 2D sinograms: shape (num_detectors, num_angles)
    - 3D sinograms: shape (nbins, num_detectors, num_angles)

    Parameters
    ----------
    sinogram : torch.Tensor
        Input sinogram tensor:
        - Shape (num_detectors, num_angles) for 2D.
        - Shape (nbins, num_detectors, num_angles) for 3D.
    angles : array-like
        Sequence of projection angles in degrees.
    output_size : int
        Desired side length of the output image or volume (assumes square).
    grid_scaled : torch.Tensor or None, optional
        Grid for resampling during backprojection (only for 3D).
        If provided, used with F.grid_sample.
    normalize : bool, optional
        If True, scale output by π / num_angles.
    device : str, optional
        Target device (default is 'cuda').

    Returns
    -------
    recon : torch.Tensor
        Reconstructed image or volume:
        - Shape (1, 1, H, W) for 2D.
        - Shape (1, nbins, H, W) for 3D.
    """
    sinogram = sinogram.to(device)
    angles = [float(a) for a in angles]

    if sinogram.ndim == 2:
        # 2D case
        num_detectors, n_angles = sinogram.shape
        recon = torch.zeros((1, 1, output_size, output_size), device=device)

        for i, theta in enumerate(angles):
            proj = sinogram[:, i]
            proj_2d = proj.view(num_detectors, 1).repeat(1, output_size).unsqueeze(0).unsqueeze(0)
            rotated = rotate(proj_2d, angle=theta + 90, interpolation=InterpolationMode.BILINEAR)
            recon += rotated

    elif sinogram.ndim == 3:
        # 3D case
        nbins, num_detectors, n_angles = sinogram.shape
        recon = torch.zeros((1, nbins, output_size, output_size), device=device)

        for i, theta in enumerate(angles):
            proj = sinogram[:, :, i]  # shape: (nbins, num_detectors)
            proj = proj.unsqueeze(1)  # shape: (nbins, 1, num_detectors)

            # Expand along width
            proj_2d = proj.repeat(1, output_size, 1)  # (nbins, output_size, num_detectors)
            proj_2d = proj_2d.permute(0, 2, 1)        # (nbins, num_detectors, output_size)
            proj_2d = proj_2d.unsqueeze(1)            # (nbins, 1, H, W)

            # Rotate to backproject
            proj_rot = rotate(proj_2d, angle=theta + 90, interpolation=InterpolationMode.BILINEAR)

            if grid_scaled is not None:
                proj_rot = proj_rot.unsqueeze(0)
                proj_rot = F.grid_sample(proj_rot, grid_scaled, mode='bilinear', align_corners=True)
                recon += proj_rot[0]
            else:
                recon += proj_rot

    else:
        raise ValueError("sinogram must be 2D or 3D (nbins, n_detectors, n_angles)")

    if normalize:
        recon *= torch.pi / len(angles)

    return recon

def sirt_pytorch(sinogram, angles, output_size, n_iter=20, relax=1.0, epsilon=1e-6, device='cuda'):
    """
    SIRT reconstruction using PyTorch with function-based forward and backward projectors.

    Parameters
    ----------
    sinogram : torch.Tensor
        Input sinogram of shape (num_detectors, num_angles), on GPU.
    angles : array-like
        Projection angles in degrees.
    output_size : int
        Output image size (assumed square).
    n_iter : int
        Number of SIRT iterations.
    relax : float
        Relaxation parameter.
    epsilon : float
        Small constant to avoid division by zero.
    device : str
        CUDA device string, e.g., 'cuda'.

    Returns
    -------
    x : torch.Tensor
        Reconstructed image of shape (1, 1, output_size, output_size)
    """
    # Sensitivity map
    W_voxel = backproject_torch(
        forwardproject_torch(torch.ones((output_size, output_size), device=device), angles, device=device),
        angles, output_size, normalize=False, device=device
    )

    # Initial guess
    x = torch.zeros((1, 1, output_size, output_size), device=device)

    for _ in range(n_iter):
        residual = sinogram - forwardproject_torch(x[0, 0], angles, device=device)
        correction = backproject_torch(residual, angles, output_size, normalize=False, device=device)
        x += relax * correction / (W_voxel + epsilon)

    return x

def cgls_pytorch(sinogram, angles, output_size, n_iter=10, device='cuda'):
    """
    CGLS solver using PyTorch and functional forward/back projectors.

    Parameters
    ----------
    sinogram : torch.Tensor
        Input sinogram of shape (num_detectors, num_angles), on GPU.
    angles : array-like
        Projection angles in degrees.
    output_size : int
        Size of the reconstructed image (assumed square).
    n_iter : int
        Number of CGLS iterations.
    device : str
        CUDA device string.

    Returns
    -------
    x : torch.Tensor
        Reconstructed image of shape (1, 1, output_size, output_size)
    """
    def forward(x): return forwardproject_torch(x[0, 0], angles, device=device)
    def backward(y): return backproject_torch(y, angles, output_size, normalize=False, device=device)

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
