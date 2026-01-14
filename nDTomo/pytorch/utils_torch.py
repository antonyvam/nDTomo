# -*- coding: utf-8 -*-
"""
Utility functions for patch-based sampling, index generation, total variation (TV) regularization, 
and structural similarity (SSIM) loss in 2D and 3D tensor volumes. These are commonly used in image 
reconstruction, inverse problems, and spectral imaging applications.

Author: Antony Vamvakeros

Contents:
- Index generation utilities:
    * generate_indices: Draws (row, col) indices from uniform, normal, or Sobol distributions.
    * generate_sobol_indices_batch: Batch generation of Sobol-distributed indices.
    * draw_valid_indices: Wrapper for drawing valid indices using masks and custom sampling.
    * filter_patch_indices: Filters mask into valid patch locations.
    * calc_patches_indices: Extracts tensor patches based on index positions.
    * initialize_counter, update_counter: Track pixel utilization in sampling-based methods.

- Normalization:
    * denormalize: Converts normalized parameter values [0, 1] back to their physical scale.

- Total Variation (TV) regularization:
    * tv_spatial: TV along spatial dimensions (y, x) of a 2D image.
    * tv_spectral: TV along spectral (channel) dimension.
    * tv_3d_spectral: TV across both spectral and spatial dimensions (channel, y, x).
    * tv_3d: TV in full 3D volumes (z, y, x), supporting 5D tensors (B, C, D, H, W).

- Structural Similarity (SSIM) loss functions:
    * SSIM2D: Perceptual similarity loss for 2D images based on luminance, contrast, and structure.
    * SSIM3DLoss: SSIM loss adapted for 3D volumes using a uniform kernel.

Notes:
All functions are implemented with PyTorch and designed to run on either CPU or CUDA. 
The TV and SSIM losses are differentiable and can be integrated directly into model training loops 
for regularization or perceptual fidelity in reconstruction tasks.
"""

import torch
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
from torch import nn
import numpy as np


def denormalize(param, param_name, param_min, param_max, peak_number=None):

    """
    Denormalize the parameter from [0, 1] to its original range.
    
    Parameters:
    param (float): The parameter to be denormalized.
    param_name (str): The name of the parameter.
    param_min (dict): A dictionary containing the minimum values of the parameters.
    param_max (dict): A dictionary containing the maximum values of the parameters.
    peak_number (int, optional): The peak number for the parameter. Default is None.
    
    Returns:
    float: The denormalized parameter.
    """
        
    if peak_number is None:  # For Slope and Intercept
        return param_min[param_name] + (param_max[param_name] - param_min[param_name]) * param
    return param_min[param_name][peak_number] + (param_max[param_name][peak_number] - param_min[param_name][peak_number]) * param



def calc_patches_indices(indices, tensor, patch_size, use_middle=False):
    """
    Calculate the indices of the patches to be selected based on the provided indices.
    Users can choose to use the middle of the patch or the top-left as the reference point.
    
    Parameters:
    indices (list of tuples): A list of tuples containing the starting indices of the patches.
    tensor (torch.Tensor): The tensor from which the patches are to be selected.
    patch_size (int): The size of the patches.
    use_middle (bool): If True, use the middle pixel as the reference. If False, use the top-left.
    
    Returns:
    torch.Tensor: A tensor containing the selected patches.
    """
    selected_patches = []
    if use_middle:
        half_patch = int(patch_size / 2)
        for (h_start, w_start) in indices:
            h_center = h_start - half_patch
            w_center = w_start - half_patch
            patch = tensor[..., h_center:h_center + patch_size, w_center:w_center + patch_size]
            selected_patches.append(patch)
    else:
        for (h_start, w_start) in indices:
            patch = tensor[..., h_start:h_start + patch_size, w_start:w_start + patch_size]
            selected_patches.append(patch)

    # Concatenate patches for processing
    return torch.cat(selected_patches, dim=0)


def generate_sobol_indices_batch(rows, cols, batch_size, patch_size, device):
    """
    Generate a batch of Sobol indices for a given dimension and batch size, considering the patch size to avoid boundary issues.
    
    Parameters:
    rows (int): The total number of rows in the dataset.
    cols (int): The total number of columns in the dataset.
    batch_size (int): The number of indices to generate.
    patch_size (int): The size of the patch to consider for boundary adjustment.
    device (torch.device): The device to use for generating indices.
    
    Returns:
    tuple: Two tensors containing the row indices and column indices.
    """
    sobol = SobolEngine(dimension=2, scramble=True)

    # Define the range for indices to avoid boundary issues
    max_row = rows - patch_size
    max_col = cols - patch_size

    # Generate indices on the CPU
    points = sobol.draw(batch_size)
    
    # Move the generated points to the specified device and scale them
    points = points.to(device) * torch.tensor([max_row, max_col], device=device, dtype=torch.float32)
    points = points.int()

    return points[:, 0], points[:, 1]


def generate_indices(rows, cols, num_indices, patch_size, distribution_type='uniform', std=3, mask=None, device='cuda', batched=False):

    """
    Generate a specified number of valid (row, col) indices from a 2D grid, optionally constrained by a mask 
    and following a given sampling distribution.

    Parameters
    ----------
    rows : int
        Number of rows in the 2D grid.
    cols : int
        Number of columns in the 2D grid.
    num_indices : int
        Total number of valid (row, col) pairs to generate.
    patch_size : int
        Size of the patch, used only for Sobol sampling mode.
    distribution_type : str, optional
        Sampling distribution to use: 'uniform', 'normal', or 'Sobol'. Default is 'uniform'.
    std : float, optional
        Standard deviation for 'normal' distribution sampling. Ignored for other distributions. Default is 3.
    mask : torch.Tensor or None, optional
        A binary mask (shape: [rows, cols]) where only non-zero locations are considered valid. Default is None.
    device : str, optional
        Device to perform computations on ('cuda' or 'cpu'). Default is 'cuda'.
    batched : bool, optional
        If True, samples are drawn in batches to speed up sampling. Default is False.

    Returns
    -------
    indices : list of tuple[int, int]
        A list of (row, col) tuples representing valid sampled locations, possibly mask-constrained.
    """
    
    mean_row = float(rows) / 2
    mean_col = float(cols) / 2
    indices = []
    
    while len(indices) < num_indices:
        batch_size = num_indices - len(indices) if batched else 1

        if distribution_type == 'normal':
            sample_rows = torch.normal(mean=torch.full((batch_size,), mean_row, dtype=torch.float32, device=device),
                                       std=torch.full((batch_size,), std, dtype=torch.float32, device=device)).round().int()
            sample_cols = torch.normal(mean=torch.full((batch_size,), mean_col, dtype=torch.float32, device=device),
                                       std=torch.full((batch_size,), std, dtype=torch.float32, device=device)).round().int()
        elif distribution_type == 'uniform':
            sample_rows = torch.randint(0, rows, (batch_size,), device=device)
            sample_cols = torch.randint(0, cols, (batch_size,), device=device)
        elif distribution_type == 'Sobol':
            sample_rows, sample_cols = generate_sobol_indices_batch(rows, cols, batch_size, patch_size, device)
        else:
            raise ValueError("Invalid distribution type specified. Choose 'normal', 'uniform', or 'Sobol'.")

        sample_rows = sample_rows.long()  # Ensure indices are long for masking
        sample_cols = sample_cols.long()

        # Validate indices within range
        valid = (sample_rows >= 0) & (sample_rows < rows) & (sample_cols >= 0) & (sample_cols < cols)

        # Apply mask if available and ensure indices are used for mask access are already validated
        if mask is not None and valid.any():
            valid_indices = valid.nonzero(as_tuple=True)
            valid_mask = mask[sample_rows[valid_indices], sample_cols[valid_indices]]
            # Refine valid to include only those indices where mask is true
            valid[valid_indices] = valid_mask

        # Collect valid indices
        valid_sample_rows = sample_rows[valid]
        valid_sample_cols = sample_cols[valid]

        indices.extend(zip(valid_sample_rows.tolist(), valid_sample_cols.tolist()))
        if len(indices) >= num_indices:
            indices = indices[:num_indices]  # Ensure we do not exceed the number of requested indices

    return indices


def initialize_counter(rows, cols):
    """
    Initialize a zero-filled counter matrix of the same size as the input matrix using PyTorch.
    
    Parameters:
    - matrix (2D torch tensor): Matrix whose dimensions will be used to create the counter.
    
    Returns:
    - 2D torch tensor: Initialized counter matrix.
    """
    return torch.zeros((rows, cols), dtype=torch.float32)

def update_counter(counter, indices, patch_size=(1, 1)):
    """
    Update the pixel utilization counter based on the middle indices of extracted patches or pixels using PyTorch.
    
    Parameters:
    - counter (2D torch tensor): Counter matrix to be updated.
    - indices (list of tuples): Indices of the top-left corners of the patches or pixels.
    - patch_size (tuple): Size (height, width) of the patches or single pixel (default is 1x1 for pixels).
    """
    patch_rows, patch_cols = patch_size
    for r, c in indices:
        # counter[r - int(patch_rows/2):r + int(patch_rows/2), c - int(patch_cols/2):c + int(patch_cols/2)] += 1
        counter[r:r + int(patch_rows), c :c + int(patch_cols)] += 1



def draw_valid_indices(rows, cols, num_indices, patch_size, distribution_type='normal', 
                       batched=False, std_dev=3, mask=None, device='cuda'):

    """
    Draw valid indices from a specified distribution.
    
    Parameters:
    rows (int): The total number of rows.
    cols (int): The total number of columns.
    num_indices (int): The number of indices to draw.
    patch_size (int): The size of the square patch.
    distribution_type (str): The type of distribution to draw from ('normal', 'uniform' or 'Sobol').
    std_dev (float): The standard deviation for the normal distribution.
    mask (torch.Tensor): A mask to apply to the indices.
    device (torch.device): The device on which to generate the indices.
    
    Returns:
    list: A list of valid indices.
    """
    
    if distribution_type == 'normal' and std_dev is None:
        largest_dimension = max(rows, cols)
        std_dev = largest_dimension / 3  # 3 sigma to span the largest dimension
    indices = generate_indices(rows, cols, num_indices, patch_size, distribution_type=distribution_type, 
                               std=std_dev, mask=mask, device=device, batched=batched)
    return indices


def filter_patch_indices(mask, patch_size):
    
    """
    Divide a binary mask into non-overlapping patches and return the top-left coordinates 
    of patches that contain any non-zero elements.

    Parameters
    ----------
    mask : torch.Tensor
        A 2D binary tensor indicating valid regions (non-zero entries).
    patch_size : int
        Size of the square patch to extract.

    Returns
    -------
    patch_indices : list of tuple[int, int]
        A list of (row, col) coordinates indicating the top-left corners of valid patches.
        The list is randomly shuffled.
    """
        
    # Generate indices for patches, filtering based on mask
    patch_indices = []
    for i in range(0, mask.shape[0], patch_size):
        for j in range(0, mask.shape[1], patch_size):
            mask_patch = mask[i:i + patch_size, j:j + patch_size]
            # Check if the mask patch is all zeros
            if not torch.all(mask_patch == 0):
                patch_indices.append((i, j))

    # Shuffle the indices randomly
    if patch_indices:  # Check if list is not empty
        idx_shuffle = torch.randperm(len(patch_indices))
        patch_indices = [patch_indices[i] for i in idx_shuffle]

    return patch_indices




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

    div_x = torch.cat((grad_x[:, :, :, :1], 
                        grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1], 
                        -grad_x[:, :, :, -1:]), dim=3)

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
	
class SSIM2D(torch.nn.Module):
    
    """
    A PyTorch module for computing the Structural Similarity Index (SSIM) between two 2D images.
    SSIM is a perceptual metric that captures similarity in terms of luminance, contrast, and structure.

    This implementation supports single-channel 2D images (e.g. grayscale) and uses a 2D Gaussian
    filter window for local statistics computation.

    Parameters
    ----------
    window_size : int, optional
        Size of the Gaussian filter window. Default is 11.
    sigma : float, optional
        Standard deviation of the Gaussian kernel. Default is 1.5.
    C1 : float, optional
        Stabilizing constant for luminance term. Default is (0.01)^2.
    C2 : float, optional
        Stabilizing constant for contrast term. Default is (0.03)^2.
    device : str, optional
        Device to store the Gaussian window tensor. Default is 'cuda'.

    Forward
    -------
    img1 : torch.Tensor
        First image tensor of shape (N, 1, H, W).
    img2 : torch.Tensor
        Second image tensor of shape (N, 1, H, W).

    Returns
    -------
    loss : torch.Tensor
        A scalar tensor representing 1 - SSIM(img1, img2). Suitable for use as a loss function.

    Notes
    -----
    - The output is `1 - SSIM`, so this module can be used directly as a loss function in optimization.
    - Assumes input tensors are normalized to [0, 1].
    - Only supports single-channel inputs (channel = 1).
    """
        
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2, device='cuda'):
        super(SSIM2D, self).__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.channel = 1  # Adjust if using more channels

        # Define Gaussian window
        window = self.create_window(window_size, sigma)
        self.window = window.to(device)

    @staticmethod
    def gaussian(window_size, sigma):
        # Create a tensor from 0 to window_size
        x = torch.arange(window_size).float() - (window_size - 1) / 2
        # Calculate the Gaussian function
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        # Normalize to ensure the sum is 1
        gauss = gauss / gauss.sum()
        return gauss

    def create_window(self, window_size, sigma):
        # Create 1D Gaussian window
        g1d = self.gaussian(window_size, sigma)
        # Use outer product to create 2D window
        g2d = torch.outer(g1d, g1d)
        # Add batch and channel dimensions [1, 1, H, W]
        window = g2d.unsqueeze(0).unsqueeze(0)
        return window

    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        SSIM_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return SSIM_map

    def forward(self, img1, img2):
        ssim_map = self._ssim(img1, img2)
        return 1 - ssim_map.mean()
    
class SSIM3DLoss(nn.Module):

    """
    Computes the Structural Similarity Index (SSIM) loss between two 3D volumes.
    SSIM is a perceptual similarity metric that evaluates structural fidelity based on local
    statistics (mean, variance, and covariance).

    This implementation uses a uniform 3D window (not Gaussian) to approximate local statistics,
    and is suitable for comparing volumes in tasks such as CT reconstruction, MRI denoising, or
    3D image synthesis.

    Parameters
    ----------
    window_size : int, optional
        Size of the cubic window used for computing local means and variances. Default is 11.

    Forward
    -------
    x : torch.Tensor
        First input volume of shape (D, H, W).
    y : torch.Tensor
        Second input volume of shape (D, H, W).

    Returns
    -------
    loss : torch.Tensor
        Scalar tensor representing `1 - SSIM(x, y)`, suitable for optimization as a loss function.

    Notes
    -----
    - Input volumes are expected to be normalized to [0, 1].
    - Adds singleton batch and channel dimensions internally.
    - Assumes single-channel input; multi-channel support would require modification.
    - The window is uniform (box filter) rather than Gaussian for simplicity.
    """
        
    def __init__(self, window_size=11):
        super(SSIM3DLoss, self).__init__()
        self.window = self.create_3D_window(window_size).cuda()  # Remove .cuda() if running on CPU
        self.window_size = window_size

    def create_3D_window(self, window_size):
        window = torch.ones(1, 1, window_size, window_size, window_size)
        return window / window.numel()

    def forward(self, x, y):
        # Add singleton dimensions for batch and channel
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)

        mu_x = F.conv3d(x, self.window, padding=self.window_size // 2, groups=1)
        mu_y = F.conv3d(y, self.window, padding=self.window_size // 2, groups=1)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x_sq = F.conv3d(x * x, self.window, padding=self.window_size // 2, groups=1) - mu_x_sq
        sigma_y_sq = F.conv3d(y * y, self.window, padding=self.window_size // 2, groups=1) - mu_y_sq
        sigma_xy  = F.conv3d(x * y, self.window, padding=self.window_size // 2, groups=1) - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        # Remove singleton dimensions
        return 1 - ssim_map.mean()