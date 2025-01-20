# -*- coding: utf-8 -*-
"""
Losses for pytorch

@author: Antony Vamvakeros

"""
#%%

import torch
import torch.nn.functional as F


def tv_spatial(x, isotropic=True, epsilon=1e-6):
    """
    Compute Total Variation (TV) gradient along spatial dimensions (y, x).

    Args:
        x: Input volume of shape (1, nch, npix, npix).
        isotropic: If True, use isotropic TV. If False, use anisotropic TV.
        epsilon: Small value to prevent division by zero (for isotropic TV).

    Returns:
        TV gradient of the same shape as x.
    """
    # Compute spatial gradients
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]  # Difference along y-axis
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]  # Difference along x-axis

    # Pad gradients to match original size
    grad_y = F.pad(grad_y, (0, 0, 1, 0))  # Pad along y-dimension
    grad_x = F.pad(grad_x, (1, 0, 0, 0))  # Pad along x-dimension

    if isotropic:
        # Isotropic: Gradient magnitude
        grad_norm = torch.sqrt(grad_y**2 + grad_x**2 + epsilon)
        grad_y /= grad_norm
        grad_x /= grad_norm
    else:
        # Anisotropic: No normalization
        grad_y = torch.sign(grad_y)
        grad_x = torch.sign(grad_x)

    # Compute divergence
    div_y = torch.cat((grad_y[:, :, :1, :], 
                        grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :], 
                        -grad_y[:, :, -1:, :]), dim=2)
    div_y = div_y[:, :, :199, :]  # Trim to match original size

    div_x = torch.cat((grad_x[:, :, :, :1], 
                        grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1], 
                        -grad_x[:, :, :, -1:]), dim=3)
    div_x = div_x[:, :, :, :199]  # Trim to match original size

    # Total variation gradient
    tv_grad = div_y + div_x

    return tv_grad


def tv_spectral(x, isotropic=True, epsilon=1e-6):
    """
    Compute Total Variation (TV) gradient along the spectral dimension (nch).

    Args:
        x: Input volume of shape (1, nch, npix, npix).
        isotropic: If True, use isotropic TV. If False, use anisotropic TV.
        epsilon: Small value to prevent division by zero (for isotropic TV).

    Returns:
        TV gradient of the same shape as x.
    """
    # Compute spectral gradient
    grad_spec = x[:, 1:, :, :] - x[:, :-1, :, :]  # Difference along spectral (nch) dimension


    if isotropic:
        # Isotropic: Gradient magnitude
        grad_norm = torch.sqrt(grad_spec**2 + epsilon)
        grad_spec = grad_spec / (grad_norm + epsilon)  # Normalize gradient
    else:
        # Anisotropic: No normalization
        grad_spec = torch.sign(grad_spec)

    # Compute divergence
    div_spec = torch.cat((grad_spec[:, :1, :, :], 
                          grad_spec[:, 1:, :, :] - grad_spec[:, :-1, :, :], 
                          -grad_spec[:, -1:, :, :]), dim=1)

    return div_spec


def tv_3d_spectral(x, isotropic=True, epsilon=1e-6):
    """
    Compute Total Variation (TV) gradient along spectral and spatial dimensions (nch, y, x).

    Args:
        x: Input volume of shape (1, nch, npix, npix).
        isotropic: If True, use isotropic TV. If False, use anisotropic TV.
        epsilon: Small value to prevent division by zero (for isotropic TV).

    Returns:
        TV gradient of the same shape as x.
    """
    # Compute gradients
    grad_spec = x[:, 1:, :, :] - x[:, :-1, :, :]  # Spectral gradient (nch dimension)
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]  # Spatial gradient along y-axis
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]  # Spatial gradient along x-axis

    # Pad gradients to match the original size
    grad_spec = F.pad(grad_spec, (0, 0, 0, 0, 1, 0))  # Pad along nch dimension
    grad_y = F.pad(grad_y, (0, 0, 1, 0))  # Pad along y-dimension
    grad_x = F.pad(grad_x, (1, 0, 0, 0))  # Pad along x-dimension

    if isotropic:
        # Isotropic: Gradient magnitude
        grad_norm = torch.sqrt(grad_spec**2 + grad_y**2 + grad_x**2 + epsilon)
        grad_spec /= grad_norm
        grad_y /= grad_norm
        grad_x /= grad_norm
    else:
        # Anisotropic: No normalization
        grad_spec = torch.sign(grad_spec)
        grad_y = torch.sign(grad_y)
        grad_x = torch.sign(grad_x)

    # Compute divergences
    div_spec = F.pad(grad_spec[:, 1:, :, :] - grad_spec[:, :-1, :, :], (0, 0, 0, 0, 1, 0))  # Spectral divergence
    div_y = F.pad(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :], (0, 0, 1, 0))  # y-axis divergence
    div_x = F.pad(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1], (1, 0, 0, 0))  # x-axis divergence

    # Total variation gradient
    tv_grad = div_spec + div_y + div_x

    return tv_grad
	

def tv_3d(x, isotropic=True, epsilon=1e-6):
    """
    Compute Total Variation (TV) gradient for 3D volumes along all three axes (x, y, z).

    Args:
        x: Input volume of shape (1, nch, D, H, W).
        isotropic: If True, use isotropic TV. If False, use anisotropic TV.
        epsilon: Small value to prevent division by zero (for isotropic TV).

    Returns:
        TV gradient of the same shape as x.
    """
    # Compute 3D gradients
    grad_z = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]  # Difference along z-axis
    grad_y = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]  # Difference along y-axis
    grad_x = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]  # Difference along x-axis

    # Pad gradients to match the original size
    grad_z = F.pad(grad_z, (0, 0, 0, 0, 1, 0))  # Pad along z-dimension
    grad_y = F.pad(grad_y, (0, 0, 1, 0, 0, 0))  # Pad along y-dimension
    grad_x = F.pad(grad_x, (1, 0, 0, 0, 0, 0))  # Pad along x-dimension

    if isotropic:
        # Isotropic: Gradient magnitude
        grad_norm = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + epsilon)
        grad_z /= grad_norm
        grad_y /= grad_norm
        grad_x /= grad_norm
    else:
        # Anisotropic: No normalization
        grad_z = torch.sign(grad_z)
        grad_y = torch.sign(grad_y)
        grad_x = torch.sign(grad_x)

    # Compute divergences
    div_z = F.pad(grad_z[:, :, 1:, :, :] - grad_z[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))  # Divergence along z
    div_y = F.pad(grad_y[:, :, :, 1:, :] - grad_y[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0))  # Divergence along y
    div_x = F.pad(grad_x[:, :, :, :, 1:] - grad_x[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0))  # Divergence along x

    # Total variation gradient
    tv_grad = div_z + div_y + div_x

    return tv_grad
	
