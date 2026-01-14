# -*- coding: utf-8 -*-
"""
PyTorch functions for 2D affine image registration, volume warping, and point cloud alignment.

This module provides differentiable implementations for aligning 2D images and 3D volumes
using geometric transformations. It includes tools for both intensity-based image registration
and geometry-based point cloud registration (ICP).

Main features:
--------------
- Differentiable 2D and 3D registration (`register_affine_2d` and `register_affine_3d`) supporting Rotation, Translation, Scale, and Shear.
- Batched volume warping (`warp_volume_xy_batched`) to apply 2D transforms to 3D stacks efficiently.
- Point cloud alignment using a differentiable Iterative Closest Point (`icp_torch`) implementation.
- Utilities for handling pixel-space vs. normalized-space affine matrices.

Author: Antony Vamvakeros
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==============================================================================
#                               HELPER UTILITIES
# ==============================================================================

def to_tensor2d(x, device=None, dtype=torch.float32):
    """
    Converts a 2D NumPy array or Tensor into a 4D Torch Tensor (1, 1, H, W).
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if x.dim() == 2:    # H,W
        x = x.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return x.to(device=device, dtype=dtype)

def to_tensor3d(x, device=None, dtype=torch.float32):
    """
    Converts a 3D NumPy array or Tensor into a 5D Torch Tensor (1, 1, D, H, W).
    """
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    # Check dimensions
    if x.dim() == 3:    # D, H, W
        x = x.unsqueeze(0).unsqueeze(0)  # 1, 1, D, H, W
    elif x.dim() == 4:  # C, D, H, W (e.g. grayscale channel included)
        x = x.unsqueeze(0)
        
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return x.to(device=device, dtype=dtype)
    
def normalize_affine_matrix(matrix_pixel, height, width):
    """
    Converts a 3x3 affine matrix defined in pixel coordinates (e.g., from skimage or pystackreg)
    into the normalized coordinate system [-1, 1] required by PyTorch.
    
    Parameters
    ----------
    matrix_pixel : numpy.ndarray
        3x3 affine matrix in pixel units.
    height, width : int
        Dimensions of the image.
        
    Returns
    -------
    torch.Tensor
        (1, 2, 3) tensor ready for F.affine_grid.
    """
    norm_mat = np.array([
        [2 / width,      0, -1],
        [0,      2 / height, -1],
        [0,           0,  1]
    ])
    # T_norm = N * T_pix * N^-1
    tm_norm = norm_mat @ matrix_pixel @ np.linalg.inv(norm_mat)
    # Extract top 2 rows
    return torch.tensor(tm_norm[:2, :], dtype=torch.float32).unsqueeze(0)

# ==============================================================================
#                               LOSS FUNCTIONS
# ==============================================================================

class NCC(torch.nn.Module):
    """
    Zero-Normalized Cross Correlation (robust to brightness changes).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        xm, ym = x.mean(), y.mean()
        xv, yv = x - xm, y - ym
        num = (xv * yv).sum()
        den = torch.sqrt((xv * xv).sum() * (yv * yv).sum() + self.eps)
        return 1 - num / (den + self.eps)

def mae_loss(x, y):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(x - y))

def mse_loss(x, y):
    """Mean Squared Error"""
    return torch.mean((x - y) ** 2)
    


# ==============================================================================
#                          IMAGE REGISTRATION (2D)
# ==============================================================================

def build_affine_matrix(theta, tx, ty, sx, sy, shx, shy, device, dtype):
    """
    Constructs the 3x3 Inverse Affine Matrix (normalized coordinates) combining 
    rotation, translation, scale, and shear.
    """
    # 1. Translation
    T = torch.eye(3, device=device, dtype=dtype)
    T[0, 2] = tx
    T[1, 2] = ty

    # 2. Rotation
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.eye(3, device=device, dtype=dtype)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c

    # 3. Shear
    Sh = torch.eye(3, device=device, dtype=dtype)
    Sh[0, 1] = shx
    Sh[1, 0] = shy

    # 4. Scale
    S = torch.eye(3, device=device, dtype=dtype)
    S[0, 0] = sx
    S[1, 1] = sy

    # Combine: M_forward = T @ R @ Sh @ S
    M_fwd = T @ R @ Sh @ S
    
    # Invert for affine_grid (Output -> Input mapping)
    M_inv = torch.linalg.inv(M_fwd)
    
    return M_inv[:2, :].unsqueeze(0) 

def register_affine_2d(ref, mov, 
                       order=['rot', 'trans', 'scale', 'shear'],
                       loss_type='ncc',
                       iters=200, lr=1e-2, verbose=False, device=None):
    """
    Register two 2D images using optimization of affine parameters.

    Parameters
    ----------
    ref, mov : (H,W) numpy array or torch tensor
    order : list of str
        Parameters to optimize: 'rot', 'trans', 'scale', 'shear'.
    loss_type : str
        Objective function: 'ncc', 'mae', or 'mse'.
    iters : int
        Number of iterations.
    lr : float
        Learning rate.
    
    Returns
    -------
    warped_image : (H,W) numpy array
    params : dict
        Optimized parameters (theta, tx, ty, sx, sy, etc.)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref = to_tensor2d(ref, device=device)
    mov = to_tensor2d(mov, device=device, dtype=ref.dtype)

    # --- Initialize Parameters ---
    theta = torch.tensor(0.0, device=device, dtype=ref.dtype)
    if 'rot' in order: theta.requires_grad_(True)

    tx = torch.tensor(0.0, device=device, dtype=ref.dtype)
    ty = torch.tensor(0.0, device=device, dtype=ref.dtype)
    if 'trans' in order: 
        tx.requires_grad_(True); ty.requires_grad_(True)

    sx = torch.tensor(1.0, device=device, dtype=ref.dtype)
    sy = torch.tensor(1.0, device=device, dtype=ref.dtype)
    if 'scale' in order:
        sx.requires_grad_(True); sy.requires_grad_(True)

    shx = torch.tensor(0.0, device=device, dtype=ref.dtype)
    shy = torch.tensor(0.0, device=device, dtype=ref.dtype)
    if 'shear' in order:
        shx.requires_grad_(True); shy.requires_grad_(True)

    params = [p for p in [theta, tx, ty, sx, sy, shx, shy] if p.requires_grad]
    
    if not params:
        return mov.squeeze().cpu().numpy(), {}

    opt = torch.optim.Adam(params, lr=lr)
    
    # Select Loss
    if loss_type == 'ncc':
        loss_fn = NCC()
    elif loss_type == 'mae':
        loss_fn = mae_loss
    elif loss_type == 'mse':
        loss_fn = mse_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # --- Optimization Loop ---
    for i in tqdm(range(iters)):
        opt.zero_grad()
        
        A = build_affine_matrix(theta, tx, ty, sx, sy, shx, shy, device, ref.dtype)
        
        # align_corners=False preserves rotation center better
        grid = F.affine_grid(A, size=ref.shape, align_corners=False)
        warped = F.grid_sample(mov, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        L = loss_fn(warped, ref)
        L.backward()
        opt.step()

        if verbose and (i % max(1, iters // 5) == 0 or i == iters-1):
            print(f"[{i+1}/{iters}] Loss={L.item():.5f}")

    return warped[0,0].detach().cpu().numpy(), {
        "theta": float(theta.detach().cpu()),
        "tx": float(tx.detach().cpu()),
        "ty": float(ty.detach().cpu()),
        "sx": float(sx.detach().cpu()),
        "sy": float(sy.detach().cpu()),
        "shx": float(shx.detach().cpu()),
        "shy": float(shy.detach().cpu()),
    }


# ==============================================================================
#                       VOLUME OPERATIONS (Batched)
# ==============================================================================

def warp_volume_xy_batched(volume_np, affine_matrix, is_pixel_space=False, batch_size=32, device=None):
    """
    Apply a single 2D affine transform to every Z-slice of a 3D volume using batches.

    Parameters
    ----------
    volume_np : numpy.ndarray
        Input 3D volume (Z, H, W).
    affine_matrix : numpy.ndarray
        3x3 Affine matrix.
    is_pixel_space : bool, optional
        If True, assumes `affine_matrix` is in pixel coordinates (e.g. from Skimage) 
        and normalizes it. If False, assumes it is already PyTorch-compatible [-1, 1].
    batch_size : int
        Number of slices to process simultaneously.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    numpy.ndarray
        Transformed 3D volume (Z, H, W).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    Z, H, W = volume_np.shape
    volume_out = np.zeros_like(volume_np, dtype=np.float32)

    # Prepare Matrix
    if is_pixel_space:
        A = normalize_affine_matrix(affine_matrix, H, W).to(device)
    else:
        # Assuming input is already a 2x3 or 3x3 normalized matrix
        if affine_matrix.shape == (3,3):
             A = torch.from_numpy(affine_matrix[:2, :].astype(np.float32)).unsqueeze(0).to(device)
        else:
             A = torch.from_numpy(affine_matrix.astype(np.float32)).to(device)
             if A.dim() == 2: A = A.unsqueeze(0)

    # Processing Loop
    for i in range(0, Z, batch_size):
        bZ = min(batch_size, Z - i)
        batch = volume_np[i:i + bZ].astype(np.float32)
        
        # Shape: [Batch, Channel=1, H, W]
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device) 
        
        # Expand matrix to match batch size: [Batch, 2, 3]
        A_batch = A.repeat(bZ, 1, 1)

        grid = F.affine_grid(A_batch, batch_tensor.shape, align_corners=False)
        warped = F.grid_sample(batch_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        volume_out[i:i + bZ] = warped.squeeze(1).cpu().numpy()

    return volume_out


# ==============================================================================
#                        POINT CLOUD REGISTRATION (ICP)
# ==============================================================================

def icp_torch(A, B, max_iterations=2000, lr=0.01, tolerance=1e-6, verbose=False):
    """
    Perform rigid Iterative Closest Point (ICP) registration from point cloud B to A.
    
    Optimizes rotation (R) and translation (t) such that B_aligned = B @ R.T + t fits A.

    Parameters
    ----------
    A, B : torch.Tensor or numpy.ndarray
        Point clouds of shape (N, 3).
    max_iterations : int
        Maximum optimization steps.
    lr : float
        Learning rate.
    tolerance : float
        Convergence threshold.

    Returns
    -------
    B_aligned : numpy.ndarray
        Aligned point cloud.
    R : numpy.ndarray
        3x3 Rotation matrix.
    t : numpy.ndarray
        Translation vector.
    """
    # Convert to tensor if needed
    if not torch.is_tensor(A): A = torch.from_numpy(A).float()
    if not torch.is_tensor(B): B = torch.from_numpy(B).float()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A, B = A.to(device), B.to(device)

    # Initialize Rigid Parameters
    R_eye = torch.eye(3, device=device)
    t_vec = torch.zeros(3, device=device)
        
    R = torch.eye(3, device=device, requires_grad=True)
    t = torch.zeros(3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([R, t], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    prev_loss = float('inf')

    for i in range(max_iterations):
        optimizer.zero_grad()
        
        # Apply transform - Note: Usually points are row vectors (N,3), so B @ R.T + t is correct
        B_transformed = B @ R.T + t
        
        # Loss: For ICP we need nearest neighbors. 
        # If A and B correspond point-to-point (same order), we use MSE.
        # If they are unorganized, we assume correspondence for this snippet.
        distances = torch.norm(B_transformed - A, dim=1)
        loss = distances.mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if abs(prev_loss - loss.item()) < tolerance:
            if verbose: print(f"Converged at iter {i}")
            break
        prev_loss = loss.item()

    # Orthogonalize R post-optimization (Procrustes projection) to ensure it's a valid rotation
    with torch.no_grad():
        u, s, v = torch.svd(R)
        R_ortho = u @ v.T
        B_final = (B @ R_ortho.T + t).cpu().numpy()
        
    return B_final, R_ortho.cpu().numpy(), t.cpu().numpy()
    
# ==============================================================================
#                          VOLUME REGISTRATION (3D)
# ==============================================================================

def build_affine_matrix_3d(rx, ry, rz, tx, ty, tz, sx, sy, sz, device, dtype):
    """
    Constructs the 3x4 Inverse Affine Matrix for 3D volumetric transformations.
    
    Combines Translation, Rotation (Euler angles XYZ), and Scaling.
    Shear is omitted for simplicity but can be added if needed.

    Parameters
    ----------
    rx, ry, rz : torch.Tensor
        Rotation angles around X, Y, and Z axes (in radians).
    tx, ty, tz : torch.Tensor
        Translation in x, y, z (normalized coordinates [-1, 1]).
    sx, sy, sz : torch.Tensor
        Scale factors for x, y, z.
    
    Returns
    -------
    torch.Tensor
        (1, 3, 4) tensor representing the top three rows of the 4x4 inverse 
        affine matrix, suitable for 3D `F.affine_grid`.
    """
    # 1. Translation (4x4)
    T = torch.eye(4, device=device, dtype=dtype)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz

    # 2. Scaling (4x4)
    S = torch.eye(4, device=device, dtype=dtype)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz

    # 3. Rotations (4x4)
    # Rotation around X
    Rx = torch.eye(4, device=device, dtype=dtype)
    cx, sx_ = torch.cos(rx), torch.sin(rx)
    Rx[1, 1], Rx[1, 2] = cx, -sx_
    Rx[2, 1], Rx[2, 2] = sx_, cx

    # Rotation around Y
    Ry = torch.eye(4, device=device, dtype=dtype)
    cy, sy_ = torch.cos(ry), torch.sin(ry)
    Ry[0, 0], Ry[0, 2] = cy, sy_
    Ry[2, 0], Ry[2, 2] = -sy_, cy

    # Rotation around Z
    Rz = torch.eye(4, device=device, dtype=dtype)
    cz, sz_ = torch.cos(rz), torch.sin(rz)
    Rz[0, 0], Rz[0, 1] = cz, -sz_
    Rz[1, 0], Rz[1, 1] = sz_, cz

    # Combine Rotations: R = Rz * Ry * Rx (standard Euler order)
    R = Rz @ Ry @ Rx

    # Full Forward Matrix: M = T * R * S
    M_fwd = T @ R @ S
    
    # Invert for affine_grid (Output -> Input mapping)
    M_inv = torch.linalg.inv(M_fwd)
    
    # Return top 3 rows (3x4 matrix) for 3D affine_grid
    return M_inv[:3, :].unsqueeze(0)
    
def register_affine_3d(ref, mov, 
                       order=['rot', 'trans', 'scale'],
                       loss_type='ncc',
                       iters=200, lr=1e-2, verbose=False, device=None):
    """
    Register two 3D volumes using optimization of affine parameters.

    Parameters
    ----------
    ref, mov : (D,H,W) numpy array or torch tensor
        Input volumes.
    order : list of str
        Parameters to optimize: 'rot', 'trans', 'scale'.
    loss_type : str
        Objective function: 'ncc', 'mae', or 'mse'.
    iters : int
        Number of iterations.
    lr : float
        Learning rate.
    
    Returns
    -------
    warped_vol : (D,H,W) numpy array
        The registered moving volume.
    params : dict
        Optimized parameters (rx, ry, rz, tx, ty, tz, sx, sy, sz).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref = to_tensor3d(ref, device=device)
    mov = to_tensor3d(mov, device=device, dtype=ref.dtype)

    # --- Initialize Parameters ---
    
    # Rotations (radians)
    rx = torch.tensor(0.0, device=device, dtype=ref.dtype)
    ry = torch.tensor(0.0, device=device, dtype=ref.dtype)
    rz = torch.tensor(0.0, device=device, dtype=ref.dtype)
    if 'rot' in order:
        rx.requires_grad_(True)
        ry.requires_grad_(True)
        rz.requires_grad_(True)

    # Translations (x, y, z)
    tx = torch.tensor(0.0, device=device, dtype=ref.dtype)
    ty = torch.tensor(0.0, device=device, dtype=ref.dtype)
    tz = torch.tensor(0.0, device=device, dtype=ref.dtype)
    if 'trans' in order: 
        tx.requires_grad_(True)
        ty.requires_grad_(True)
        tz.requires_grad_(True)

    # Scales (sx, sy, sz)
    sx = torch.tensor(1.0, device=device, dtype=ref.dtype)
    sy = torch.tensor(1.0, device=device, dtype=ref.dtype)
    sz = torch.tensor(1.0, device=device, dtype=ref.dtype)
    if 'scale' in order:
        sx.requires_grad_(True)
        sy.requires_grad_(True)
        sz.requires_grad_(True)

    # Collect active parameters
    params_list = [p for p in [rx, ry, rz, tx, ty, tz, sx, sy, sz] if p.requires_grad]
    
    if not params_list:
        print("Warning: No parameters selected for optimization.")
        return mov.squeeze().cpu().numpy(), {}

    opt = torch.optim.Adam(params_list, lr=lr)
    
    # Select Loss
    if loss_type == 'ncc':
        loss_fn = NCC()
    elif loss_type == 'mae':
        loss_fn = mae_loss
    elif loss_type == 'mse':
        loss_fn = mse_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # --- Optimization Loop ---
    for i in tqdm(range(iters)):
        opt.zero_grad()
        
        # Build 3D Affine Matrix
        A = build_affine_matrix_3d(rx, ry, rz, tx, ty, tz, sx, sy, sz, device, ref.dtype)
        
        # Grid Sample 3D
        # Note: 5D grid_sample expects (N, C, D, H, W)
        grid = F.affine_grid(A, size=ref.shape, align_corners=False)
        warped = F.grid_sample(mov, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        L = loss_fn(warped, ref)
        L.backward()
        opt.step()

        if verbose and (i % max(1, iters // 5) == 0 or i == iters-1):
            print(f"[{i+1}/{iters}] Loss={L.item():.5f}")

    return warped.squeeze().detach().cpu().numpy(), {
        "rx": float(rx.detach().cpu()), "ry": float(ry.detach().cpu()), "rz": float(rz.detach().cpu()),
        "tx": float(tx.detach().cpu()), "ty": float(ty.detach().cpu()), "tz": float(tz.detach().cpu()),
        "sx": float(sx.detach().cpu()), "sy": float(sy.detach().cpu()), "sz": float(sz.detach().cpu()),
    }