# -*- coding: utf-8 -*-
"""
Methods for creating geometric shapes in 3D.

This module provides a collection of functions for generating and manipulating 
various 3D geometric shapes within NumPy arrays. The implemented shapes range 
from basic primitives such as spheres, cubes, and cylinders to more complex 
structures like toroids, Möbius strips, and fractal-based geometries.

Each function operates on a 3D NumPy array, modifying it by filling the specified 
shape with a given value. The methods utilize vectorized operations for efficiency 
and support both solid and hollow versions of certain shapes where applicable.

These functions are useful for synthetic data generation, volumetric modeling, 
scientific simulations, and 3D image processing tasks.

@author: Antony Vamvakeros
"""

import numpy as np
from tqdm import tqdm

def create_sphere(arr, center, outer_radius, thickness=0, fill_value=1):
    """
    Creates a solid or hollow sphere in a 3D array.

    This function generates either a solid sphere or a hollow spherical shell
    (hollow sphere) in the given 3D array `arr`. If `thickness` is 0, a solid sphere 
    is created. Otherwise, a shell is formed where voxels within the specified 
    thickness are filled.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the sphere is drawn.
    center : tuple of int
        (x0, y0, z0) coordinates specifying the center of the sphere.
    outer_radius : int or float
        The outer radius of the sphere.
    thickness : int or float, optional
        Thickness of the shell (default is 0, which creates a solid sphere).
    fill_value : int or float, optional
        The value to fill inside the sphere or shell (default is 1).

    Returns
    -------
    arr : np.ndarray
        The modified array with the sphere or shell.
    """
    x0, y0, z0 = center
    inner_radius = max(outer_radius - thickness, 0)  # Avoid negative radius

    X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    dist2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2

    outer_r2 = outer_radius**2
    inner_r2 = inner_radius**2

    mask = dist2 <= outer_r2 if thickness == 0 else (dist2 <= outer_r2) & (dist2 >= inner_r2)
    arr[mask] = fill_value
    return arr

def create_cube(arr, center, size, thickness=0, fill_value=1):
    """
    Creates a solid or hollow cube in a 3D NumPy array.

    This function generates either a solid cube or a hollow cube shell
    in the given 3D array `arr`. If `thickness` is 0, a solid cube is created.
    Otherwise, a hollow cube with specified wall thickness is formed.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the cube will be drawn.
    center : tuple of int
        (x, y, z) coordinates of the cube's center.
    size : int
        The edge length of the cube.
    thickness : int, optional
        The thickness of the cube walls (default is 0, which creates a solid cube).
    fill_value : int or float, optional
        The value to fill inside the cube or its walls (default is 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the cube or hollow cube drawn.
    """
    x0, y0, z0 = center
    half_size = size // 2

    # Define cube boundaries
    x_min, x_max = max(x0 - half_size, 0), min(x0 + half_size, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_size, 0), min(y0 + half_size, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_size, 0), min(z0 + half_size, arr.shape[2] - 1)

    # Fill the cube region
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value

    # If thickness > 0, create a hollow cube by removing the inner region
    if thickness > 0:
        inner_x_min, inner_x_max = x_min + thickness, x_max - thickness
        inner_y_min, inner_y_max = y_min + thickness, y_max - thickness
        inner_z_min, inner_z_max = z_min + thickness, z_max - thickness

        # Ensure we don't completely remove the cube
        if inner_x_min < inner_x_max and inner_y_min < inner_y_max and inner_z_min < inner_z_max:
            arr[inner_x_min:inner_x_max+1, inner_y_min:inner_y_max+1, inner_z_min+1:inner_z_max] = 0

    return arr

def create_cuboid(arr, center, size_x, size_y, size_z, thickness=0, fill_value=1):
    """
    Creates a solid or hollow cuboid in a 3D NumPy array.

    This function generates either a solid cuboid or a hollow cuboid shell
    in the given 3D array `arr`. If `thickness` is 0, a solid cuboid is created.
    Otherwise, a hollow cuboid with specified wall thickness is formed.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the cuboid will be drawn.
    center : tuple of int
        (x, y, z) coordinates of the cuboid's center.
    size_x : int
        The length of the cuboid along the x-axis.
    size_y : int
        The length of the cuboid along the y-axis.
    size_z : int
        The length of the cuboid along the z-axis.
    thickness : int, optional
        The thickness of the cuboid walls (default is 0, which creates a solid cuboid).
    fill_value : int or float, optional
        The value to fill inside the cuboid or its walls (default is 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the cuboid or hollow cuboid drawn.
    """
    x0, y0, z0 = center
    half_x, half_y, half_z = size_x // 2, size_y // 2, size_z // 2

    # Define cuboid boundaries
    x_min, x_max = max(x0 - half_x, 0), min(x0 + half_x, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_y, 0), min(y0 + half_y, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_z, 0), min(z0 + half_z, arr.shape[2] - 1)

    # Fill the cuboid region
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value

    # If thickness > 0, create a hollow cuboid by removing the inner region
    if thickness > 0:
        inner_x_min, inner_x_max = x_min + thickness, x_max - thickness
        inner_y_min, inner_y_max = y_min + thickness, y_max - thickness
        inner_z_min, inner_z_max = z_min + thickness, z_max - thickness

        # Ensure we don't completely remove the cuboid
        if inner_x_min < inner_x_max and inner_y_min < inner_y_max and inner_z_min < inner_z_max:
            arr[inner_x_min:inner_x_max+1, inner_y_min:inner_y_max+1, inner_z_min+1:inner_z_max] = 0

    return arr


def create_cylinder(arr, center_xy, z_range, outer_radius, thickness=1, fill_value=1, caps=True):
    """
    Creates a solid or hollow cylinder in a 3D NumPy array, oriented along the Z-axis.

    The function supports both solid and hollow cylinders. If `thickness=0`, a solid 
    cylinder is created. If `thickness > 0`, a hollow cylinder with a specified wall 
    thickness is created. Optional top and bottom caps can be included.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the cylinder will be drawn.
    center_xy : tuple of int
        (cx, cy) coordinates of the cylinder's center in the X-Y plane.
    z_range : tuple of int
        (z_start, z_end) indices defining the height of the cylinder along the Z-axis.
    outer_radius : int
        Outer radius of the cylinder's base.
    thickness : int, optional
        Radial thickness of the cylinder walls. If set to 0 (default), a solid cylinder is created.
    fill_value : int or float, optional
        The value used to fill the cylinder or its walls.
    caps : bool, optional
        If True, the cylinder is closed off with top and bottom caps. If False, the cylinder remains open.

    Returns
    -------
    np.ndarray
        Updated 3D array with the cylinder drawn.
    """

    if thickness == 0:
        cx, cy = center_xy
        z_start, z_end = z_range
        
        # Make sure z-range is within bounds
        z_start = max(z_start, 0)
        z_end   = min(z_end, arr.shape[2])

        # Create coordinate grids
        X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
        
        # Distance in X-Y plane from (cx, cy)
        dist2_xy = (X - cx)**2 + (Y - cy)**2
        
        # Condition for being inside the cylinder cross-section
        cross_section_mask = dist2_xy <= outer_radius**2
        # Condition for being within z-range
        z_mask = (Z >= z_start) & (Z < z_end)
        
        # Combine both masks
        mask = cross_section_mask & z_mask
        arr[mask] = fill_value
        return arr

    else:
        cx, cy = center_xy
        z_start, z_end = z_range

        # Clamp z-range to the array's bounds
        z_start = max(z_start, 0)
        z_end   = min(z_end, arr.shape[2])
        if z_start >= z_end:
            return arr  # no valid Z range, do nothing

        # Define inner radius (ensure it's non-negative)
        inner_radius = max(outer_radius - thickness, 0)

        # Create coordinate arrays with ogrid
        X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

        # Distance (squared) in the X-Y plane from center (cx, cy)
        dist2_xy = (X - cx)**2 + (Y - cy)**2
        outer_r2 = outer_radius**2
        inner_r2 = inner_radius**2

        # -------------------------
        # 1) Side Walls
        # -------------------------
        if caps:
            # Side region is from z in [z_start + thickness, z_end - thickness)
            # This ensures that the caps remain visible.
            side_z_min = min(z_start + thickness, z_end)  
            side_z_max = max(z_end - thickness, z_start)  
        else:
            # If no caps, the side spans the entire [z_start, z_end)
            side_z_min = z_start
            side_z_max = z_end

        # Side walls: ring shape in cross-section,
        # for z in [side_z_min, side_z_max)
        side_mask_xy = (dist2_xy <= outer_r2) & (dist2_xy >= inner_r2)
        side_mask_z  = (Z >= side_z_min) & (Z < side_z_max)
        side_mask = side_mask_xy & side_mask_z

        # -------------------------
        # 2) Caps (only if caps=True)
        # -------------------------
        if caps:
            # -- TOP CAP --
            # Covers z in [z_end - thickness, z_end)
            top_z_min = max(z_end - thickness, z_start)
            top_mask_z = (Z >= top_z_min) & (Z < z_end)
            top_mask_xy = (dist2_xy <= outer_r2)
            top_mask = top_mask_xy & top_mask_z

            # -- BOTTOM CAP --
            # Covers z in [z_start, z_start + thickness)
            bot_z_max = min(z_start + thickness, z_end)
            bot_mask_z = (Z >= z_start) & (Z < bot_z_max)
            bot_mask_xy = (dist2_xy <= outer_r2)
            bot_mask = bot_mask_xy & bot_mask_z

            # Combine side + caps
            final_mask = side_mask | top_mask | bot_mask
        else:
            # If caps=False, only side walls
            final_mask = side_mask

        # Fill in the final region
        arr[final_mask] = fill_value
        return arr



def create_cone(arr, tip, height, outer_radius, thickness=1, fill_value=1, caps=False):
    """
    Creates a solid or hollow cone in a 3D NumPy array, oriented along the Z-axis.

    The function supports both solid and hollow cones. If `thickness=0`, a solid 
    cone is created. If `thickness > 0`, a hollow cone with a specified wall 
    thickness is created. Optional base caps can be included to close the bottom 
    of the cone.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the cone will be drawn.
    tip : tuple of int
        (x0, y0, z0) coordinates of the cone's tip.
    height : int
        The height of the cone along the Z-axis.
    outer_radius : float
        The radius of the cone at the base.
    thickness : float, optional (default=1)
        Radial thickness of the conical shell. If set to 0, a solid cone is created.
    fill_value : int or float, optional
        The value used to fill the cone or its walls.
    caps : bool, optional (default=False)
        If True, the cone is closed off at the base with a filled annular disk. 
        If False, the cone remains open at the base.

    Returns
    -------
    arr : np.ndarray
        The modified 3D array with the cone drawn.
    """

    if thickness == 0:
        x0, y0, z0 = tip
        z1 = z0 + height

        X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
        
        # Fraction of height completed at each Z
        # (Z - z0)/height goes from 0 at the tip to 1 at the base
        frac = (Z - z0) / height
        
        # Only consider 0 <= frac <= 1 (within cone's vertical extent)
        within_height_mask = (frac >= 0) & (frac <= 1)
        
        # Radius at each z-layer
        r_current = outer_radius * frac
        
        # Distance in X-Y plane from (x0, y0)
        dist2_xy = (X - x0)**2 + (Y - y0)**2
        r2_current = r_current**2  # square the radius for direct comparison
        
        # Condition for inside the cone's cross-section
        cross_section_mask = dist2_xy <= r2_current
        
        mask = within_height_mask & cross_section_mask
        arr[mask] = fill_value
        return arr

    else:

        x0, y0, z0 = tip
        z1 = z0 + height

        # Create mesh grid
        X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

        # Compute normalized height fraction for each Z-layer
        frac = (Z - z0) / height
        within_height_mask = (frac >= 0) & (frac <= 1)

        # Outer radius at each Z-layer
        r_outer = outer_radius * frac

        # Corrected **inner radius** at each Z-layer for uniform thickness
        r_inner = np.maximum(r_outer - thickness, 0)

        r2_outer = r_outer**2
        r2_inner = r_inner**2

        # Distance squared from cone axis in X-Y plane
        dist2_xy = (X - x0)**2 + (Y - y0)**2

        # Side shell: Hollow conical shape with **constant thickness**
        shell_mask = (dist2_xy <= r2_outer) & (dist2_xy >= r2_inner)
        side_mask = within_height_mask & shell_mask

        # ---------------------------------------------------------
        # Add a base cap (disk) if caps=True
        # ---------------------------------------------------------
        if caps:
            base_mask_z = (Z >= z1 - thickness) & (Z < z1)  # Fill last 'thickness' layers
            base_mask_xy = (dist2_xy <= outer_radius**2)  # Fully solid base (not just hollow)
            base_mask = base_mask_z & base_mask_xy
            final_mask = side_mask | base_mask
        else:
            final_mask = side_mask

        arr[final_mask] = fill_value
        return arr


def create_pyramid(arr, tip, height, base_size, thickness=1, fill_value=1):
    """
    Creates a solid or hollow pyramid in a 3D NumPy array, oriented along the Z-axis.

    The pyramid has a square base centered at `(x0, y0, z0 + height)`, tapering 
    to a tip at `(x0, y0, z0)`. If `thickness=0`, a solid pyramid is created. 
    If `thickness > 0`, a hollow pyramid with a uniform wall thickness is created. 
    Optional base and tip caps can be included to fully enclose the pyramid.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the pyramid will be drawn.
    tip : tuple of int
        (x0, y0, z0) coordinates of the pyramid's tip.
    height : int
        The height of the pyramid along the Z-axis.
    base_size : int
        The width of the pyramid's square base at `z0 + height`.
    thickness : int, optional (default=1)
        The thickness of the pyramid's walls. If set to 0, a solid pyramid is created.
    fill_value : int or float, optional
        The value used to fill the pyramid or its walls.
    caps : bool, optional (default=False)
        If True, adds a solid base to the pyramid.

    Returns
    -------
    arr : np.ndarray
        The modified 3D array with the pyramid drawn.
    """

    if thickness == 0:
        x0, y0, z0 = tip
        z1 = z0 + height

        X, Y, Z = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

        # Compute the fraction along Z
        frac = (Z - z0) / height
        within_height_mask = (frac >= 0) & (frac <= 1)

        # Compute the half-width of the pyramid at each height
        half_width = (base_size / 2) * frac

        # Define boundary conditions
        x_min = x0 - half_width
        x_max = x0 + half_width
        y_min = y0 - half_width
        y_max = y0 + half_width

        # Create the mask
        pyramid_mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max) & within_height_mask

        arr[pyramid_mask] = fill_value
        return arr
    else:
        x0, y0, z0 = tip
        z1 = z0 + height

        X, Y, Z = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

        # Compute the fraction along Z
        frac = (Z - z0) / height
        within_height_mask = (frac >= 0) & (frac <= 1)

        # Outer pyramid boundary
        half_width_outer = (base_size / 2) * frac
        x_min_outer = x0 - half_width_outer
        x_max_outer = x0 + half_width_outer
        y_min_outer = y0 - half_width_outer
        y_max_outer = y0 + half_width_outer

        # Inner pyramid boundary (uniform wall thickness)
        half_width_inner = np.maximum(half_width_outer - thickness, 0)
        x_min_inner = x0 - half_width_inner
        x_max_inner = x0 + half_width_inner
        y_min_inner = y0 - half_width_inner
        y_max_inner = y0 + half_width_inner

        # Create hollow pyramid mask
        pyramid_mask_outer = (X >= x_min_outer) & (X <= x_max_outer) & (Y >= y_min_outer) & (Y <= y_max_outer)
        pyramid_mask_inner = (X >= x_min_inner) & (X <= x_max_inner) & (Y >= y_min_inner) & (Y <= y_max_inner)

        # Hollow shell
        shell_mask = within_height_mask & (pyramid_mask_outer & ~pyramid_mask_inner)

        # Base cap
        base_mask = (Z >= z1 - thickness) & (Z <= z1) & pyramid_mask_outer
        # Tip cap
        tip_mask = (Z >= z0) & (Z <= z0 + thickness) & pyramid_mask_outer

        # Combine side shell + base cap + tip cap
        final_mask = shell_mask | base_mask | tip_mask    
        arr[final_mask] = fill_value
        return arr

def create_torus(arr, center, major_radius, minor_radius, fill_value=1, hollow_thickness=0):
    """
    Creates a torus (3D ring) in `arr`, centered at `center`, with major and minor radii.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the torus is drawn.
    center : tuple (x0, y0, z0)
        Center of the torus.
    major_radius : float
        Distance from the torus center to the middle of the ring.
    minor_radius : float
        Thickness of the torus tube.
    fill_value : int or float, optional
        The value to fill the torus.
    hollow_thickness : float, optional (default=0)
        If > 0, makes the torus hollow with tube thickness `hollow_thickness`.

    Returns
    -------
    arr : np.ndarray
        The modified array.
    """

    x0, y0, z0 = center
    X, Y, Z = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

    # Distance from center in XY plane
    dist_xy = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

    # Define torus equation
    torus_mask = (major_radius - dist_xy) ** 2 + (Z - z0) ** 2 <= minor_radius ** 2

    if hollow_thickness > 0:
        inner_radius = minor_radius - hollow_thickness
        torus_mask &= (major_radius - dist_xy) ** 2 + (Z - z0) ** 2 >= inner_radius ** 2

    arr[torus_mask] = fill_value
    return arr


def create_ellipsoid(arr, center, radii, fill_value=1, hollow_thickness=0):
    """
    Creates a 3D ellipsoid in `arr`, centered at `center`, with radii `(a, b, c)`.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the ellipsoid is drawn.
    center : tuple (x0, y0, z0)
        Center of the ellipsoid.
    radii : tuple (a, b, c)
        Semi-axes lengths along X, Y, Z.
    fill_value : int or float, optional
        The value to fill the ellipsoid.
    hollow_thickness : float, optional (default=0)
        If > 0, makes the ellipsoid hollow with wall thickness `hollow_thickness`.

    Returns
    -------
    arr : np.ndarray
        The modified array.
    """

    x0, y0, z0 = center
    a, b, c = radii

    X, Y, Z = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

    # Ellipsoid equation
    ellipsoid_mask = ((X - x0) ** 2 / a ** 2) + ((Y - y0) ** 2 / b ** 2) + ((Z - z0) ** 2 / c ** 2) <= 1

    if hollow_thickness > 0:
        a_inner, b_inner, c_inner = max(a - hollow_thickness, 0.1), max(b - hollow_thickness, 0.1), max(c - hollow_thickness, 0.1)
        inner_mask = ((X - x0) ** 2 / a_inner ** 2) + ((Y - y0) ** 2 / b_inner ** 2) + ((Z - z0) ** 2 / c_inner ** 2) <= 1
        ellipsoid_mask &= ~inner_mask  # Make it hollow

    arr[ellipsoid_mask] = fill_value
    return arr


def create_hexagonal_prism(arr, center_xy, z_range, outer_radius, fill_value=1):
    """
    Creates a hexagonal prism in `arr`.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the hexagonal prism is drawn.
    center_xy : tuple (cx, cy)
        Center of the hexagonal cross-section.
    z_range : tuple (z_min, z_max)
        Z-range for the prism.
    outer_radius : float
        Outer radius of the hexagon.
    fill_value : int or float, optional
        The value to fill the hexagonal prism.

    Returns
    -------
    arr : np.ndarray
        The modified array.
    """

    cx, cy = center_xy
    z_min, z_max = z_range
    X, Y, Z = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]

    qx = np.abs(X - cx)
    qy = np.abs(Y - cy)

    # Hexagon equation
    hex_mask = (qy <= outer_radius) & (qx * np.sqrt(3) + qy <= outer_radius * 2)

    # Restrict to Z-range
    z_mask = (Z >= z_min) & (Z < z_max)

    arr[hex_mask & z_mask] = fill_value
    return arr


def create_mobius_strip(arr, center, major_radius, width, num_points=100, fill_value=1):
    """
    Creates a Möbius strip in `arr` using parametric equations.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the Möbius strip is drawn.
    center : tuple (x0, y0, z0)
        Center of the Möbius strip.
    major_radius : float
        Main radius of the strip.
    width : float
        Width of the Möbius band.
    num_points : int, optional
        Resolution of the generated Möbius strip.
    fill_value : int or float, optional
        The value to fill the Möbius strip.

    Returns
    -------
    arr : np.ndarray
        The modified array.
    """

    x0, y0, z0 = center
    theta = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(-width / 2, width / 2, num_points)

    X = (major_radius + v * np.cos(theta / 2)) * np.cos(theta)
    Y = (major_radius + v * np.cos(theta / 2)) * np.sin(theta)
    Z = v * np.sin(theta / 2)

    for i in range(len(X)):
        arr[int(X[i] + x0), int(Y[i] + y0), int(Z[i] + z0)] = fill_value

    return arr


def create_menger_sponge(arr, center, size, depth, fill_value=1):
    """
    Recursively creates a 3D Menger sponge inside `arr`.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the Menger sponge is drawn.
    center : tuple (cx, cy, cz)
        Center of the sponge.
    size : int
        Side length of the current cube.
    depth : int
        Recursion depth of the sponge.
    fill_value : int or float, optional
        The value to fill the sponge.

    Returns
    -------
    arr : np.ndarray
        The modified array.
    """

    if depth == 0:
        # Base case: Fill a cube at this location
        cx, cy, cz = center
        half_size = size // 2
        x_min, x_max = cx - half_size, cx + half_size
        y_min, y_max = cy - half_size, cy + half_size
        z_min, z_max = cz - half_size, cz + half_size
        arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value
        return arr

    # Recursive case: Divide into 27 smaller cubes
    step = size // 3
    offsets = [-step, 0, step]  # Possible shifts in each direction

    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                # Skip the center cube and face centers
                if (dx == 0 and dy == 0) or (dy == 0 and dz == 0) or (dz == 0 and dx == 0):
                    continue
                
                # Compute new center and recursively draw the smaller cube
                new_center = (center[0] + dx, center[1] + dy, center[2] + dz)
                create_menger_sponge(arr, new_center, step, depth - 1, fill_value)

    return arr


def create_voronoi_3d(arr, seed_points, fill_values=None):
    """
    Creates a Voronoi diagram in a 3D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        3D array where the Voronoi diagram is drawn.
    seed_points : list of tuples [(x1, y1, z1), (x2, y2, z2), ...]
        List of seed points for the Voronoi cells.
    fill_values : list, optional
        List of values corresponding to each seed point.
        If None, assigns unique integer values to each region.

    Returns
    -------
    arr : np.ndarray
        The modified array with Voronoi regions.
    """

    D, H, W = arr.shape  # Get depth, height, width
    X, Y, Z = np.mgrid[:D, :H, :W]  # Create coordinate grids

    # Compute squared distance to each seed point
    dist_matrix = np.full((D, H, W, len(seed_points)), np.inf)

    for i, (x, y, z) in tqdm(enumerate(seed_points)):
        dist_matrix[:, :, :, i] = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2  # Squared Euclidean distance

    # Find the closest seed point for each voxel
    closest_seed = np.argmin(dist_matrix, axis=3)

    # Assign region values
    if fill_values is None:
        arr[:, :, :] = closest_seed + 1  # Assign unique values per region
    else:
        arr[:, :, :] = np.vectorize(lambda i: fill_values[i])(closest_seed)

    return arr

