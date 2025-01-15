# -*- coding: utf-8 -*-
"""
Methods for creating shapes in 2D and 3D

@author: Antony Vamvakeros
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
from nDTomo.utils.hyperexpl import ImageSpectrumGUI
from nDTomo.utils.misc3D import showvol
import time

#%%

def create_circle(arr, center, radius, fill_value=1):
    """
    Creates a solid circle in 2D array `arr` using vectorized operations.
    center = (cx, cy), radius = r.
    """
    cx, cy = center
    H, W = arr.shape
    X, Y = np.ogrid[:H, :W]  # X -> [0..H-1], Y -> [0..W-1]
    
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask = dist2 <= radius**2
    
    arr[mask] = fill_value
    return arr


def create_circle_hollow(arr, center, outer_radius, thickness=1, fill_value=1):
    """
    Creates a hollow circle (ring) in 2D array `arr`.
    Distance between (outer_radius - thickness) and outer_radius is set.
    """
    cx, cy = center
    inner_radius = max(outer_radius - thickness, 0)
    
    H, W = arr.shape
    X, Y = np.ogrid[:H, :W]
    
    dist2 = (X - cx)**2 + (Y - cy)**2
    outer_r2 = outer_radius**2
    inner_r2 = inner_radius**2
    
    mask = (dist2 <= outer_r2) & (dist2 >= inner_r2)
    arr[mask] = fill_value
    return arr


def create_rectangle_corner(arr, corner, width, height, fill_value=1):
    """
    Creates a rectangle given by its top-left corner (x0, y0) 
    and dimensions `width` (along y) and `height` (along x),
    using slicing (vectorized).
    """
    x0, y0 = corner
    H, W = arr.shape
    
    # Bound the rectangle so we don't go out of array
    x_end = min(x0 + height, H)
    y_end = min(y0 + width, W)
    
    arr[x0:x_end, y0:y_end] = fill_value
    return arr


def create_rectangle_corner_hollow(arr, corner, width, height, thickness=1, fill_value=1):
    """
    Fills only the boundary of a rectangle (hollow) 
    with top-left corner (x0, y0).
    """
    x0, y0 = corner
    H, W = arr.shape
    
    x_end = min(x0 + height, H)
    y_end = min(y0 + width, W)
    
    # Top edge
    arr[x0 : x0+thickness, y0 : y_end] = fill_value
    # Bottom edge
    arr[x_end-thickness : x_end, y0 : y_end] = fill_value
    # Left edge
    arr[x0 : x_end, y0 : y0+thickness] = fill_value
    # Right edge
    arr[x0 : x_end, y_end-thickness : y_end] = fill_value
    return arr


def create_triangle(arr, p1, p2, p3, fill_value=1):
    """
    Creates a triangle in 2D array `arr` using a vectorized
    approach with a bounding-box mask and a barycentric check.
    """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    H, W = arr.shape

    # 1) Compute bounding box of the triangle
    min_x = max(min(x1, x2, x3), 0)
    max_x = min(max(x1, x2, x3), H-1)
    min_y = max(min(y1, y2, y3), 0)
    max_y = min(max(y1, y2, y3), W-1)
    
    if min_x > max_x or min_y > max_y:
        return  # No area to fill

    # 2) Create sub-grid
    X, Y = np.ogrid[min_x:max_x+1, min_y:max_y+1]
    
    # Barycentric method
    # area of the full triangle
    def tri_area(ax, ay, bx, by, cx, cy):
        return np.abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    
    area_ABC = tri_area(x1, y1, x2, y2, x3, y3)
    
    area_PBC = tri_area(X, Y, x2, y2, x3, y3)
    area_PAC = tri_area(x1, y1, X, Y, x3, y3)
    area_PAB = tri_area(x1, y1, x2, y2, X, Y)
    
    # Points inside if sum of sub-areas = total area
    inside_mask = (area_PBC + area_PAC + area_PAB) == area_ABC
    
    arr_region = arr[min_x:max_x+1, min_y:max_y+1]
    arr_region[inside_mask] = fill_value
    return arr


def create_triangle_hollow(arr, p1, p2, p3, thickness=1, fill_value=1):
    """
    Fills only the boundary of the triangle in a 2D array `arr`.
    One way is to draw thick lines for each edge (vectorized line-drawing
    is trickier, but you could slice a thin 'rectangle' around each line).
    For demonstration, we’ll simply revert to a line-drawing approach.
    """
    # For a truly vectorized "line thickness" in 2D, one approach is
    # to compute distance from each pixel to the line segment and
    # check if it's <= thickness/2. That can be done but is more involved.
    #
    # We'll show a partial vectorization approach:
    draw_line_thick_vectorized(arr, p1, p2, thickness, fill_value)
    draw_line_thick_vectorized(arr, p2, p3, thickness, fill_value)
    draw_line_thick_vectorized(arr, p3, p1, thickness, fill_value)
    return arr


def draw_line_thick_vectorized(arr, p1, p2, thickness=1, fill_value=1):
    """
    Draws a 'thick' line in a 2D array `arr` by computing the 
    distance from each pixel to the line. If it's <= half_thick, fill it.
    This can be quite expensive for large images, but it's fully vectorized.
    """
    (x1, y1), (x2, y2) = p1, p2
    H, W = arr.shape

    # 1) Get bounding box of the line
    min_x = max(min(x1, x2) - thickness, 0)
    max_x = min(max(x1, x2) + thickness, H-1)
    min_y = max(min(y1, y2) - thickness, 0)
    max_y = min(max(y1, y2) + thickness, W-1)
    
    if min_x > max_x or min_y > max_y:
        return
    
    # 2) Create sub-grid
    X, Y = np.ogrid[min_x:max_x+1, min_y:max_y+1]
    
    # Parametric line (x1, y1)->(x2, y2)
    # Distance from point (X, Y) to line
    # = |(y2-y1)*X - (x2-x1)*Y + (x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)
    denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if denom == 0:  # degenerate line
        return
    
    dist = np.abs((y2 - y1)*X - (x2 - x1)*Y + (x2*y1 - y2*x1)) / denom
    
    # Fill mask
    half_t = thickness / 2.0
    line_mask = dist <= half_t
    
    arr_region = arr[min_x:max_x+1, min_y:max_y+1]
    arr_region[line_mask] = fill_value


def create_equilateral_triangle(arr, p1, side, fill_value=1, orientation='default'):
    """
    Creates and fills an equilateral triangle in the 2D array `arr`
    using a vectorized barycentric approach.

    Parameters
    ----------
    arr : np.ndarray (2D)
        2D array of shape (H, W) where we will draw the triangle.
    p1 : tuple of int
        Coordinates (x1, y1) for the first vertex (anchor) of the triangle.
        x1 -> row index, y1 -> column index.
    side : int or float
        The length of each side of the equilateral triangle.
    fill_value : int or float, optional
        The value used to fill inside the triangle (default=1).
    orientation : str, optional
        - 'default': Base is horizontal, apex above the base.
        - 'down'   : Base is horizontal, apex below the base.
        - 'random' : Base at random angle, apex “above” the base vector.

    Returns
    -------
    arr : np.ndarray
        The same array passed in, but with the triangle’s interior set to `fill_value`.
    """

    (x1, y1) = p1
    # Compute the "height" for an equilateral triangle
    # We'll round to int because array indices must be integers
    height = int(round((np.sqrt(3) / 2) * side))

    # Convert side to an integer if necessary
    # (in case side was float, to ensure consistent indexing)
    side_int = int(round(side))

    if orientation == 'default':
        # -------------------------------------------
        # Base is left-to-right at row x1
        # Apex is above the base
        # -------------------------------------------
        # p1 = (x1, y1)         (left corner of base)
        # p2 = (x1, y1 + side)  (right corner of base)
        # p3 is above by `height`
        p2 = (x1, y1 + side_int)
        p3 = (x1 - height, y1 + side_int // 2)

    elif orientation == 'down':
        # -------------------------------------------
        # Base is left-to-right at row x1
        # Apex is below the base
        # -------------------------------------------
        # p1 = (x1, y1)
        # p2 = (x1, y1 + side)
        # p3 is below by `height`
        p2 = (x1, y1 + side_int)
        p3 = (x1 + height, y1 + side_int // 2)

    elif orientation == 'random':
        # -------------------------------------------
        # Base is at a random angle alpha in [0, 2*pi),
        # measured in image coordinates (x=rows downward, y=columns to the right).
        # We'll interpret alpha=0 as base pointing to the right,
        # and apex "above" means in the negative-x direction from the base.
        # -------------------------------------------
        alpha = np.random.uniform(0, 2*np.pi)

        # Base vector from p1 to p2
        # side (row offset) = side * cos(alpha), side (col offset) = side * sin(alpha)
        base_dx = side * np.cos(alpha)
        base_dy = side * np.sin(alpha)

        # For an equilateral triangle, the apex is found by:
        # apex_offset = side*(sqrt(3)/2) in a direction perpendicular to the base vector
        # A 90-degree rotation of (cos alpha, sin alpha) -> (-sin alpha, cos alpha)
        # We'll put the apex "above" the base in image coordinates, i.e., negative row offset
        # so that visually it's "above" if alpha=0. 
        # apex_dx = (sqrt(3)/2 * side) * (-sin alpha)
        # apex_dy = (sqrt(3)/2 * side) * ( cos alpha)
        apex_scale = (np.sqrt(3)/2) * side
        apex_dx = apex_scale * (-np.sin(alpha))
        apex_dy = apex_scale * ( np.cos(alpha))

        # Now define p2 and p3
        x2_f = x1 + base_dx   # floating point
        y2_f = y1 + base_dy
        x3_f = x1 + (base_dx / 2.0) + apex_dx
        y3_f = y1 + (base_dy / 2.0) + apex_dy

        # Round them to int for array indexing
        p2 = (int(round(x2_f)), int(round(y2_f)))
        p3 = (int(round(x3_f)), int(round(y3_f)))

    else:
        raise ValueError("orientation must be one of ['default', 'down', 'random']")

    # Now we have p1, p2, p3. Perform a barycentric fill just like create_triangle.
    (xA, yA) = p1
    (xB, yB) = p2
    (xC, yC) = p3

    H, W = arr.shape

    # 1) Compute bounding box of the triangle
    min_x = max(min(xA, xB, xC), 0)
    max_x = min(max(xA, xB, xC), H - 1)
    min_y = max(min(yA, yB, yC), 0)
    max_y = min(max(yA, yB, yC), W - 1)

    # If bounding box is invalid or zero-area, just return
    if min_x > max_x or min_y > max_y:
        return arr

    # 2) Create sub-grid
    X, Y = np.ogrid[min_x:max_x+1, min_y:max_y+1]

    # Barycentric area helper
    def tri_area(ax, ay, bx, by, cx, cy):
        return np.abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))

    area_ABC = tri_area(xA, yA, xB, yB, xC, yC)

    # If the area is 0 (collinear points), do nothing
    if area_ABC == 0:
        return arr

    area_PBC = tri_area(X, Y, xB, yB, xC, yC)
    area_PAC = tri_area(xA, yA, X, Y, xC, yC)
    area_PAB = tri_area(xA, yA, xB, yB, X, Y)

    # Points are inside if sum of sub-areas = total area
    inside_mask = (area_PBC + area_PAC + area_PAB) == area_ABC

    # Fill the relevant region in `arr`
    arr_region = arr[min_x:max_x+1, min_y:max_y+1]
    arr_region[inside_mask] = fill_value

    return arr

def create_ellipse(arr, center, axes_radii, fill_value=1):
    """
    Creates a solid ellipse in the 2D array `arr`.
    
    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (H, W).
    center : tuple of float
        (cx, cy): center of the ellipse in (row, col) format.
    axes : tuple of float
        (a, b): the radius of the semi-major and semi-minor axes (or vice versa).
    fill_value : int or float, optional
        The value to fill inside the ellipse, default = 1.
    """
    cx, cy = center
    a, b = axes_radii  # a -> radius in x-direction (rows), b -> radius in y-direction (cols)

    H, W = arr.shape
    # Create a grid of indices
    X, Y = np.ogrid[:H, :W]

    # Equation of ellipse: ((X-cx)^2 / a^2) + ((Y-cy)^2 / b^2) <= 1
    ellipse_mask = ((X - cx)**2 / a**2) + ((Y - cy)**2 / b**2) <= 1

    arr[ellipse_mask] = fill_value
    return arr


def create_ellipse_hollow(arr, center, outer_axes_radii, thickness=1, fill_value=1):
    """
    Fills a hollow elliptical ring in 2D array `arr`.
    `outer_axes` = (a, b) => outer ellipse.
    'thickness' is subtracted from both a and b to get the inner ellipse.
    """
    cx, cy = center
    a_outer, b_outer = outer_axes_radii

    # Ensure we don’t go negative
    a_inner = max(a_outer - thickness, 0.1)  # avoid zero
    b_inner = max(b_outer - thickness, 0.1)

    H, W = arr.shape
    X, Y = np.ogrid[:H, :W]

    outer_mask = ((X - cx)**2 / a_outer**2) + ((Y - cy)**2 / b_outer**2) <= 1
    inner_mask = ((X - cx)**2 / a_inner**2) + ((Y - cy)**2 / b_inner**2) <= 1
    
    # final mask: within outer ellipse but outside inner ellipse
    ring_mask = outer_mask & (~inner_mask)
    arr[ring_mask] = fill_value
    return arr

def create_star(arr, center, n_points=5, r_outer=10, r_inner=5, fill_value=1, angle_offset=0.0):
    """
    Creates (and fills) a star in the 2D array `arr`.
    The star is built as a polygon and then filled.
    """
    cx, cy = center
    vertices = []

    # Build the star's vertices
    # 2*n_points total vertices (outer+inner)
    for k in range(2*n_points):
        theta = angle_offset + (np.pi * k / n_points)
        # Even k -> outer vertex, odd k -> inner vertex
        r = r_outer if (k % 2 == 0) else r_inner
        x = cx + r*np.cos(theta)
        y = cy + r*np.sin(theta)
        vertices.append((int(round(x)), int(round(y))))

    # Now fill polygon
    return create_polygon(arr, vertices, fill_value=fill_value)


def create_polygon(arr, vertices, fill_value=1):
    """
    Creates an arbitrary polygon in `arr` using a 2D ray-casting approach.
    This version uses np.mgrid to avoid shape-mismatch issues.
    """
    # Extract x (row) and y (col)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]

    H, W = arr.shape

    # Compute bounding box
    min_x = max(min(xs), 0)
    max_x = min(max(xs), H - 1)
    min_y = max(min(ys), 0)
    max_y = min(max(ys), W - 1)

    if min_x > max_x or min_y > max_y:
        return arr  # Nothing to fill

    # Build 2D mesh grid (shape => (#rows, #cols))
    X, Y = np.mgrid[min_x:max_x+1, min_y:max_y+1]
    # Flatten them
    XX = X.ravel()
    YY = Y.ravel()

    # Perform point-in-polygon (ray-casting) test
    inside_mask_1d = _point_in_poly_1d(XX, YY, vertices)

    # Reshape mask to same shape as X, Y
    inside_mask = inside_mask_1d.reshape(X.shape)

    # Fill the region
    arr[min_x:max_x+1, min_y:max_y+1][inside_mask] = fill_value
    return arr


def _point_in_poly_1d(xs, ys, vertices):
    """
    Vectorized "point in polygon" test (ray casting).
    xs, ys are 1D arrays of points.
    vertices is a list of polygon vertices [(x0, y0), (x1, y1), ...].
    """
    px = np.asarray(xs, dtype=float)
    py = np.asarray(ys, dtype=float)
    vs = np.array(vertices, dtype=float)
    n = len(vertices)

    x_coords = vs[:, 0]
    y_coords = vs[:, 1]
    x_next = np.roll(x_coords, -1)
    y_next = np.roll(y_coords, -1)

    crossing = np.zeros(px.shape, dtype=int)

    for i in range(n):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_next[i], y_next[i]

        # Ensure y1 <= y2 for consistency
        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # Condition A: py in (y1, y2]
        condA = (py > y1) & (py <= y2)
        if y2 == y1:
            continue  # horizontal edge => skip

        # X intersection of the line at py
        x_int = x1 + (x2 - x1)*(py - y1)/(y2 - y1)

        # Condition B: x_int >= px
        condB = (x_int >= px)

        # Combine => an intersection event
        edge_mask = condA & condB
        crossing[edge_mask] += 1

    # Inside if crossing is odd
    return (crossing % 2) == 1


def create_voronoi(arr, seed_points, fill_values=None):
    """
    Creates a Voronoi diagram in a 2D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        2D array where the Voronoi diagram is drawn.
    seed_points : list of tuples [(x1, y1), (x2, y2), ...]
        List of seed points for the Voronoi cells.
    fill_values : list, optional
        List of values corresponding to each seed point.
        If None, assigns unique integer values to each region.

    Returns
    -------
    arr : np.ndarray
        The modified array with Voronoi regions.
    """

    H, W = arr.shape
    X, Y = np.mgrid[:H, :W]

    # Compute squared distance to each seed point
    dist_matrix = np.full((H, W, len(seed_points)), np.inf)

    for i, (x, y) in enumerate(seed_points):
        dist_matrix[:, :, i] = (X - x) ** 2 + (Y - y) ** 2  # Squared Euclidean distance

    # Find the closest seed point for each pixel
    closest_seed = np.argmin(dist_matrix, axis=2)

    # Assign region colors
    if fill_values is None:
        arr[:, :] = closest_seed + 1  # Assign unique values per region
    else:
        arr[:, :] = np.vectorize(lambda i: fill_values[i])(closest_seed)

    return arr

####################### 3D Shapes #######################

def create_sphere(arr, center, radius, fill_value=1):
    """
    Fills a solid sphere of radius `radius` in 3D array `arr`,
    centered at `center = (x0, y0, z0)`.
    Vectorized approach (no explicit nested for-loops).
    """
    x0, y0, z0 = center
    X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    dist2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2

    mask = dist2 <= radius**2
    arr[mask] = fill_value
    return arr

def create_sphere_hollow(arr, center, outer_radius, thickness=1, fill_value=1):
    """
    Creates a hollow sphere shell (i.e. ring in 3D) in `arr`.
    Only voxels whose distance from `center` is between
    (outer_radius - thickness) and outer_radius are set.
    Vectorized approach.
    """
    x0, y0, z0 = center
    inner_radius = max(outer_radius - thickness, 0)

    X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    dist2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2

    outer_r2 = outer_radius**2
    inner_r2 = inner_radius**2

    mask = (dist2 <= outer_r2) & (dist2 >= inner_r2)
    arr[mask] = fill_value
    return arr

def create_cube(arr, center, size, fill_value=1):
    """
    Creates a cube in a 3D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the cube will be drawn.
    center : tuple of int
        (x, y, z) coordinates of the cube's center.
    size : int
        The edge length of the cube.
    fill_value : int or float, optional
        The value to fill inside the cube (default is 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the cube drawn.
    """
    x0, y0, z0 = center
    half_size = size // 2

    # Define cube boundaries
    x_min, x_max = max(x0 - half_size, 0), min(x0 + half_size, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_size, 0), min(y0 + half_size, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_size, 0), min(z0 + half_size, arr.shape[2] - 1)

    # Fill cube region
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value
    return arr

def create_cube_hollow(arr, center, size, thickness=1, fill_value=1):
    """
    Creates a hollow cube in a 3D NumPy array, ensuring top and bottom faces are closed.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the hollow cube will be drawn.
    center : tuple of int
        (x, y, z) coordinates of the cube's center.
    size : int
        The edge length of the cube.
    thickness : int, optional
        The thickness of the cube walls (default: 1).
    fill_value : int or float, optional
        The value to fill inside the cube walls (default: 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the hollow cube drawn.
    """
    x0, y0, z0 = center
    half_size = size // 2

    # Define cube boundaries
    x_min, x_max = max(x0 - half_size, 0), min(x0 + half_size, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_size, 0), min(y0 + half_size, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_size, 0), min(z0 + half_size, arr.shape[2] - 1)

    # Define inner hollow region
    inner_x_min, inner_x_max = x_min + thickness, x_max - thickness
    inner_y_min, inner_y_max = y_min + thickness, y_max - thickness
    inner_z_min, inner_z_max = z_min + thickness, z_max - thickness

    # Fill the outer cube first
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value

    # Hollow out the inner region but keep top and bottom layers intact
    arr[inner_x_min:inner_x_max+1, inner_y_min:inner_y_max+1, inner_z_min+1:inner_z_max] = 0
    return arr

def create_cuboid(arr, center, size_x, size_y, size_z, fill_value=1):
    """
    Creates a cuboid in a 3D NumPy array.

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
    fill_value : int or float, optional
        The value to fill inside the cuboid (default is 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the cuboid drawn.
    """
    x0, y0, z0 = center
    half_x, half_y, half_z = size_x // 2, size_y // 2, size_z // 2

    # Define cuboid boundaries
    x_min, x_max = max(x0 - half_x, 0), min(x0 + half_x, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_y, 0), min(y0 + half_y, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_z, 0), min(z0 + half_z, arr.shape[2] - 1)

    # Fill cuboid region
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value
    return arr

def create_cuboid_hollow(arr, center, size_x, size_y, size_z, thickness=1, fill_value=1):
    """
    Creates a hollow cuboid in a 3D NumPy array, ensuring top and bottom faces are closed.

    Parameters
    ----------
    arr : np.ndarray
        3D NumPy array where the hollow cuboid will be drawn.
    center : tuple of int
        (x, y, z) coordinates of the cuboid's center.
    size_x : int
        The length of the cuboid along the x-axis.
    size_y : int
        The length of the cuboid along the y-axis.
    size_z : int
        The length of the cuboid along the z-axis.
    thickness : int, optional
        The thickness of the cuboid walls (default: 1).
    fill_value : int or float, optional
        The value to fill inside the cuboid walls (default: 1).

    Returns
    -------
    np.ndarray
        Updated 3D array with the hollow cuboid drawn.
    """
    x0, y0, z0 = center
    half_x, half_y, half_z = size_x // 2, size_y // 2, size_z // 2

    # Define cuboid boundaries
    x_min, x_max = max(x0 - half_x, 0), min(x0 + half_x, arr.shape[0] - 1)
    y_min, y_max = max(y0 - half_y, 0), min(y0 + half_y, arr.shape[1] - 1)
    z_min, z_max = max(z0 - half_z, 0), min(z0 + half_z, arr.shape[2] - 1)

    # Define inner hollow region
    inner_x_min, inner_x_max = x_min + thickness, x_max - thickness
    inner_y_min, inner_y_max = y_min + thickness, y_max - thickness
    inner_z_min, inner_z_max = z_min + thickness, z_max - thickness

    # Fill the outer cuboid first
    arr[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = fill_value

    # Hollow out the inner region but keep top and bottom layers intact
    arr[inner_x_min:inner_x_max+1, inner_y_min:inner_y_max+1, inner_z_min+1:inner_z_max] = 0
    return arr


def create_cylinder(arr, center_xy, z_range, radius, fill_value=1):
    """
    Creates a solid cylinder oriented along the Z-axis in `arr`.
    `arr` is of shape (X, Y, Z).
    `center_xy = (cx, cy)`.
    `z_range = (z_start, z_end)` (these are indices along Z).
    """
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
    cross_section_mask = dist2_xy <= radius**2
    # Condition for being within z-range
    z_mask = (Z >= z_start) & (Z < z_end)
    
    # Combine both masks
    mask = cross_section_mask & z_mask
    arr[mask] = fill_value
    return arr


def create_cylinder_hollow(arr, center_xy, z_range, outer_radius, thickness=1, fill_value=1, caps=True):
    """
    Creates a hollow cylindrical shell (or pipe) oriented along the Z-axis.

    - If `caps=False` (default), it creates an open-ended hollow tube.
    - If `caps=True`, it closes off the top and bottom with ring-shaped caps of 
      thickness `thickness`, making the cylinder fully enclosed.

    Parameters
    ----------
    arr : np.ndarray, shape (X, Y, Z)
        3D array in which to draw the cylinder.
    center_xy : tuple of (float, float)
        (cx, cy) center of the cylinder cross-section in the X-Y plane.
    z_range : tuple of (int, int)
        (z_start, z_end) indices along the Z axis.
    outer_radius : float
        Outer radius of the cylinder cross-section.
    thickness : float, optional (default=1)
        Radial thickness of the walls. Also defines the thickness of the caps if `caps=True`.
    fill_value : int or float, optional
        Value used to fill the cylinder (and caps).
    caps : bool, optional (default=False)
        If False, leaves the ends open. If True, adds ring-shaped caps at the top and bottom.

    Returns
    -------
    arr : np.ndarray
        The modified array (also updated in-place).
    """

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

def create_cone(arr, tip, height, base_radius, fill_value=1):
    """
    Creates a solid right circular cone in `arr`, oriented along +Z.
    The tip is at (x0, y0, z0). The base is at z0+height.
    For z in [z0, z0+height], radius scales from 0 to base_radius.
    """
    x0, y0, z0 = tip
    z1 = z0 + height

    X, Y, Z = np.ogrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]]
    
    # Fraction of height completed at each Z
    # (Z - z0)/height goes from 0 at the tip to 1 at the base
    frac = (Z - z0) / height
    
    # Only consider 0 <= frac <= 1 (within cone's vertical extent)
    within_height_mask = (frac >= 0) & (frac <= 1)
    
    # Radius at each z-layer
    r_current = base_radius * frac
    
    # Distance in X-Y plane from (x0, y0)
    dist2_xy = (X - x0)**2 + (Y - y0)**2
    r2_current = r_current**2  # square the radius for direct comparison
    
    # Condition for inside the cone's cross-section
    cross_section_mask = dist2_xy <= r2_current
    
    mask = within_height_mask & cross_section_mask
    arr[mask] = fill_value
    return arr



def create_cone_hollow(arr, tip, height, outer_radius, thickness=1, fill_value=1, caps=False):
    """
    Creates a hollow conical shell in `arr`, oriented along +Z.

    The tip is at `(x0, y0, z0)`, and the base radius is `outer_radius`.
    The shell thickness is **uniform**, meaning the inner surface is correctly scaled.

    If `caps=True`, a ring cap is added at the base (a filled annular disk).

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape (X, Y, Z) where the hollow cone will be created.
    tip : tuple (x0, y0, z0)
        Coordinates of the tip of the cone.
    height : int
        Height of the cone along the +Z axis.
    outer_radius : float
        Outer radius of the cone at the base.
    thickness : float, optional (default=1)
        Radial thickness of the conical shell.
    fill_value : int or float, optional (default=1)
        The value used to fill the shell (and base cap if applicable).
    caps : bool, optional (default=False)
        If False, the cone remains open at the base.
        If True, a ring-shaped cap is added at the base to fully enclose it.

    Returns
    -------
    arr : np.ndarray
        The same array, modified in-place.
    """

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



def create_pyramid(arr, tip, height, base_size, fill_value=1):
    """
    Creates a solid pyramid in `arr`, oriented along +Z.

    The tip is at `(x0, y0, z0)`, and the base is a square 
    centered at `(x0, y0)` with width `base_size` at `z0 + height`.

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape (X, Y, Z) where the pyramid will be created.
    tip : tuple (x0, y0, z0)
        Coordinates of the tip of the pyramid.
    height : int
        Height of the pyramid along the +Z axis.
    base_size : int
        The width of the pyramid's base at `z0 + height` (square base).
    fill_value : int or float, optional (default=1)
        The value used to fill the pyramid.

    Returns
    -------
    arr : np.ndarray
        The same array, modified in-place.
    """

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



def create_pyramid_hollow(arr, tip, height, base_size, thickness=1, fill_value=1):
    """
    Creates a hollow pyramid shell in `arr`, oriented along +Z.

    The tip is at `(x0, y0, z0)`, and the base is a square centered 
    at `(x0, y0)` with width `base_size` at `z0 + height`.

    If `caps=True`, a base cap is added to fully enclose it.

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape (X, Y, Z) where the pyramid will be created.
    tip : tuple (x0, y0, z0)
        Coordinates of the tip of the pyramid.
    height : int
        Height of the pyramid along the +Z axis.
    base_size : int
        The width of the pyramid's base at `z0 + height` (square base).
    thickness : int, optional (default=1)
        The thickness of the pyramid's walls.
    fill_value : int or float, optional (default=1)
        The value used to fill the pyramid shell.
    caps : bool, optional (default=False)
        If True, adds a solid base to the pyramid.

    Returns
    -------
    arr : np.ndarray
        The same array, modified in-place.
    """

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

    for i, (x, y, z) in enumerate(seed_points):
        dist_matrix[:, :, :, i] = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2  # Squared Euclidean distance

    # Find the closest seed point for each voxel
    closest_seed = np.argmin(dist_matrix, axis=3)

    # Assign region values
    if fill_values is None:
        arr[:, :, :] = closest_seed + 1  # Assign unique values per region
    else:
        arr[:, :, :] = np.vectorize(lambda i: fill_values[i])(closest_seed)

    return arr


#%%



# im = np.zeros((256, 256), dtype='float32')
# imn = create_circle(im, center=(50,50), radius=10, fill_value=2)
# imn = create_circle(imn, center=(25,25), radius=20, fill_value=1)
# imn = create_circle_hollow(im, center=(100,100), outer_radius=20, thickness=4, fill_value=2)
# imn = create_rectangle_corner(im, corner=(50,50), width=10, height=30, fill_value=2)
# imn = create_rectangle_corner_hollow(im, corner=(50,50), width=10, height=30, thickness=2, fill_value=2)
# imn = create_triangle(im, p1=(100,100), p2=(120,125), p3=(65,110), fill_value=1)
# imn = create_triangle_hollow(im, p1=(100,100), p2=(120,125), p3=(65,110), thickness=2, fill_value=1)
# imn = create_equilateral_triangle(im, p1=(100,100), side=50, fill_value=1, orientation='random')
# imn = create_ellipse(im, center=(100,100), axes_radii=(20, 10), fill_value=1)
# imn = create_ellipse_hollow(im, center=(100,100), outer_axes_radii=(20, 10), thickness=1, fill_value=1)
# imn = create_star(im, center=(100,100), n_points=7, r_outer=26, r_inner=12, 
#                   fill_value=1, angle_offset=0.0)


# Define random seed points
size = (256, 256)
img = np.zeros(size, dtype=int)
np.random.seed(42)
num_points = 30
seed_points = [(np.random.randint(0, size[0]), np.random.randint(0, size[1])) for _ in range(num_points)]

# Generate Voronoi diagram
imn = create_voronoi(img, seed_points)

plt.figure(1);plt.clf()
plt.imshow(imn, cmap='gray', interpolation='None')
plt.colorbar()
plt.show()



#%%


# Create a 3D array of zeros
shape = (256, 256, 256)  # (x, y, z)
vol = np.zeros(shape, dtype='float32')

# vol = create_sphere(vol, center=(100,100,100), radius=50, fill_value=1)
# vol = create_sphere_hollow(vol, center=(100,100,100), outer_radius=50, thickness = 5, fill_value=1)
vol = create_cylinder(vol, center_xy=(99, 99), z_range = (40, 161), radius=50, fill_value=1)
# vol = create_cylinder_hollow(vol, center_xy=(99, 99), z_range=(40, 161), 
#                              outer_radius=50, thickness=5, fill_value=1, caps=True)

# vol = create_cone(vol, tip=(50,50,50), height=75, base_radius=100, fill_value=1)
# vol = create_cone_hollow(vol, tip=(100,100,50), height=75, outer_radius=100, thickness = 10, 
                        #  fill_value=1, caps=True)
# vol = create_pyramid(vol, tip=(100,100,50), height=75, base_size=100, fill_value=1)
# vol = create_pyramid_hollow(vol, tip=(100,100,50), height=75, base_size=100, fill_value=1)

# vol = create_torus(vol, center=(100,100,100), major_radius=75, minor_radius=24, fill_value=1, hollow_thickness=5)
# vol = create_ellipsoid(vol, center=(100,100,100), radii=(50, 90, 75), fill_value=1, hollow_thickness=0)

# vol = create_hexagonal_prism(vol, center_xy=(100,100), z_range=(40,160), outer_radius=10, fill_value=1)
# vol = create_mobius_strip(vol, center=(100,100,100), major_radius=100, width=100, num_points=10000, fill_value=1)
# vol = create_menger_sponge(vol, center=(100,100,100), size=100, depth=3, fill_value=1)

# num_points = 250
# seed_points = [(np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])) for _ in range(num_points)]

# # Generate 3D Voronoi diagram
# start = time.time()
# vol = create_voronoi_3d(vol, seed_points)
# print(f"Time taken: {time.time() - start:.2f} seconds")


#%%

from nDTomo.utils.misc3D import showvol

# ImageSpectrumGUI(vol)
showvol(vol, opacity_mode = 'binary', thr = 0.1)


#%%


from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
import time
import numpy as np
import os
import matplotlib.pyplot as plt

def cmap_to_ctf(cmap_name, vmin=0, vmax=1):
    """Convert a Matplotlib colormap to a Mayavi ColorTransferFunction."""
    values = np.linspace(vmin, vmax, 256)
    cmap = plt.colormaps.get_cmap(cmap_name)(values)  # Updated for Matplotlib 3.7+
    ctf = ColorTransferFunction()
    for i, v in enumerate(values):
        ctf.add_rgb_point(v, cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return ctf

def create_opacity_transfer_function(vmin, vmax):
    """Create an Opacity Transfer Function (OTF) to fix transparency issues."""
    otf = PiecewiseFunction()
    otf.add_point(vmin, 0.0)   # Fully transparent at min value
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.2)
    otf.add_point(vmin + (vmax - vmin) * 0.5, 0.5)
    otf.add_point(vmin + (vmax - vmin) * 0.8, 0.8)
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    return otf

def create_adaptive_opacity_function(vol, vmin, vmax):
    """Create an adaptive Opacity Transfer Function based on volume intensity distribution."""
    otf = PiecewiseFunction()
    
    # Compute histogram to understand intensity distribution
    hist, bin_edges = np.histogram(vol, bins=10, range=(vmin, vmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Define opacity dynamically based on intensity distribution
    for i, bin_value in enumerate(bin_centers):
        opacity = min(0.2 + hist[i] * 5, 1.0)  # Scale by histogram density
        otf.add_point(bin_value, opacity)

    return otf

def create_balanced_opacity_function(vmin, vmax):
    """Create a more uniform Opacity Transfer Function to prevent full transparency in regions."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.05)   # Almost transparent at min value
    otf.add_point(vmin + (vmax - vmin) * 0.25, 0.2)
    otf.add_point(vmin + (vmax - vmin) * 0.5, 0.4)
    otf.add_point(vmin + (vmax - vmin) * 0.75, 0.7)
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    
    return otf


def create_corrected_opacity_function(vmin, vmax):
    """Create an Opacity Transfer Function that ensures full grains are visible."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.1)   # Slight transparency at min value
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.6)  # Less transparency
    otf.add_point(vmin + (vmax - vmin) * 0.4, 0.8)  # Mostly visible
    otf.add_point(vmin + (vmax - vmin) * 0.6, 0.9)  # Almost fully visible
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    
    return otf


def create_uniform_opacity_function(vmin, vmax):
    """Create an Opacity Transfer Function that keeps grains fully visible."""
    otf = PiecewiseFunction()

    otf.add_point(vmin, 0.0)   # Fully transparent at lowest values
    otf.add_point(vmin + (vmax - vmin) * 0.1, 0.8)  # Almost fully visible at low intensity
    otf.add_point(vmax, 1.0)   # Fully opaque at max intensity


def create_solid_opacity_function():
    """Create an Opacity Transfer Function that makes everything solid (fully opaque)."""
    otf = PiecewiseFunction()
    otf.add_point(0.0, 1.0)  # Fully opaque at minimum value
    otf.add_point(1.0, 1.0)  # Fully opaque at maximum value
    return otf


def create_fade_opacity_function(vmin, vmax):
    """Smoothly fade low-intensity values instead of hard transparency."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.0)   # Fully transparent at zero
    otf.add_point(vmin + (vmax - vmin) * 0.05, 0.2)  # Slightly visible
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.6)  # More visible
    otf.add_point(vmax, 1.0)   # Fully opaque at max intensity
    
    return otf

def create_binary_opacity_function(threshold):
    """Creates an opacity function where values below `threshold` are transparent, 
    and values above it are fully opaque."""
    otf = PiecewiseFunction()

    otf.add_point(threshold - 1e-6, 0.0)  # Just before the threshold → fully transparent
    otf.add_point(threshold, 1.0)  # At threshold → fully opaque
    otf.add_point(1.0, 1.0)  # Everything else remains fully visible

    return otf

def showvol(vol, vlim=None, colormap="jet", show_axes=True, show_colorbar=True):
    '''
    Volume rendering using Mayavi mlab with customization options.
    
    Parameters:
        vol (np.ndarray): 3D volume data.
        vlim (tuple): (vmin, vmax) for intensity scaling.
        colormap (str): Colormap to use (e.g., "jet", "viridis", "gray").
        show_axes (bool): Whether to display the coordinate axes.
        show_colorbar (bool): Whether to show a colorbar.
    '''
    if vlim is None:
        vmin = 0
        vmax = np.max(vol)
    else:
        vmin, vmax = vlim
    
    # ✅ Ensure the figure is managed by mlab
    fig = mlab.gcf()  # Get the current figure

    # ✅ Create volume rendering explicitly linked to the figure
    src = mlab.pipeline.scalar_field(vol, figure=fig)
    volume = mlab.pipeline.volume(src, vmin=vmin, vmax=vmax, figure=fig)
    # volume._volume_property.interpolation_type = 'nearest'  # Try 'linear' or 'nearest'
    # volume._volume_mapper.sample_distance = 0.5  # Reduce sample distance

    # ✅ Convert colormap to ColorTransferFunction and apply it
    ctf = cmap_to_ctf(colormap, vmin, vmax)
    volume._volume_property.set_color(ctf)
    volume._ctf = ctf
    volume.update_ctf = True  # ✅ Force update

    # # ✅ Apply Adaptive Opacity Transfer Function
    # # otf = create_binary_opacity_function(0.05)
    # # otf = create_fade_opacity_function(vmin, vmax)
    otf = create_adaptive_opacity_function(vol, vmin, vmax)
    volume._volume_property.set_scalar_opacity(otf)
    volume._otf = otf
    volume.update_ctf = True  # ✅ Force update again

    # ✅ Extract and apply the same LUT to the colorbar
    lut_manager = volume.module_manager.scalar_lut_manager
    lut_manager.lut_mode = colormap  # Ensure the colorbar follows the colormap

    # ✅ Show colorbar
    if show_colorbar:
        mlab.colorbar(orientation="vertical", title="Intensity")

    # ✅ Toggle axes visibility
    if show_axes:
        mlab.orientation_axes()
    else:
        mlab.axes(visible=False)

    # return volume

# Define angles and output directory
angles = np.linspace(0, 360, 25)  # 60 frames (for example)
output_dir = "C:\\Users\\Antony\\Documents\\test\\"
os.makedirs(output_dir, exist_ok=True)

# Volume data (example)
shape = (64, 64, 64)
vol = np.zeros(shape, dtype='float32')
optimal_distance = max(shape) * 3  # ✅ Set a fixed distance instead of "auto"
# Compute the center of the volume
cx, cy, cz = np.array(shape) / 2  # ✅ Compute focal point

# vol = np.random.rand(*shape)  # Replace with your actual 3D volume
# # vol[:,:, 30:40] = 0
# vol[:,:, 30:40] = 0.01

num_points = 20
seed_points = [(np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])) for _ in range(num_points)]
# Generate 3D Voronoi diagram
start = time.time()
vol = create_voronoi_3d(vol, seed_points)
print(f"Time taken: {time.time() - start:.2f} seconds")
vol = vol - np.min(vol)
vol = (vol/np.max(vol))
vol[:1, :, :] = 0
vol[-1:, :, :] = 0
vol[:, :1, :] = 0
vol[:, -1:, :] = 0


# Initialize a single Mayavi figure
mlab.figure(bgcolor=(1, 1, 1))

# Loop through angles
for i, angle in enumerate(angles):

    print(angle)

    # Open new figure, render volume
    # fig = showvol(vol)
    showvol(vol, colormap="viridis", show_axes=True, show_colorbar=True)

    # ✅ Apply rotation (fix for freezing issue)
    mlab.view(azimuth=angle, elevation=90, distance=optimal_distance, focalpoint=(cx, cy, cz))

    # Save screenshot
    screenshot_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    mlab.savefig(screenshot_path)
    print(f"Saved: {screenshot_path}")

# Close the figure to avoid multiple windows
mlab.close()


#%%

from numpy import max, linspace, histogram
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
import matplotlib.pyplot as plt

def cmap_to_ctf(cmap_name, vmin=0, vmax=1):
    """Convert a Matplotlib colormap to a Mayavi ColorTransferFunction."""
    values = linspace(vmin, vmax, 256)
    cmap = plt.colormaps.get_cmap(cmap_name)(values)  # Updated for Matplotlib 3.7+
    ctf = ColorTransferFunction()
    for i, v in enumerate(values):
        ctf.add_rgb_point(v, cmap[i, 0], cmap[i, 1], cmap[i, 2])
    return ctf
	

def create_adaptive_opacity_function(vol, vmin, vmax):
    """Create an adaptive Opacity Transfer Function based on volume intensity distribution."""
    otf = PiecewiseFunction()
    
    # Compute histogram to understand intensity distribution
    hist, bin_edges = histogram(vol, bins=10, range=(vmin, vmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

    # Define opacity dynamically based on intensity distribution
    for i, bin_value in enumerate(bin_centers):
        opacity = min(0.2 + hist[i] * 5, 1.0)  # Scale by histogram density
        otf.add_point(bin_value, opacity)

    return otf
	
	
def create_balanced_opacity_function(vmin, vmax):
    """Create a more uniform Opacity Transfer Function to prevent full transparency in regions."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.05)   # Almost transparent at min value
    otf.add_point(vmin + (vmax - vmin) * 0.25, 0.2)
    otf.add_point(vmin + (vmax - vmin) * 0.5, 0.4)
    otf.add_point(vmin + (vmax - vmin) * 0.75, 0.7)
    otf.add_point(vmax, 1.0)   # Fully opaque at max value
    
    return otf


def create_solid_opacity_function():
    """Create an Opacity Transfer Function that makes everything solid (fully opaque)."""
    otf = PiecewiseFunction()
    otf.add_point(0.0, 1.0)  # Fully opaque at minimum value
    otf.add_point(1.0, 1.0)  # Fully opaque at maximum value
    return otf

def create_fade_opacity_function(vmin, vmax):
    """Smoothly fade low-intensity values instead of hard transparency."""
    otf = PiecewiseFunction()
    
    otf.add_point(vmin, 0.0)   # Fully transparent at zero
    otf.add_point(vmin + (vmax - vmin) * 0.05, 0.2)  # Slightly visible
    otf.add_point(vmin + (vmax - vmin) * 0.2, 0.6)  # More visible
    otf.add_point(vmax, 1.0)   # Fully opaque at max intensity
    
    return otf

def create_binary_opacity_function(threshold):
    """Creates an opacity function where values below `threshold` are transparent, 
    and values above it are fully opaque."""
    otf = PiecewiseFunction()

    otf.add_point(threshold - 1e-6, 0.0)  # Just before the threshold → fully transparent
    otf.add_point(threshold, 1.0)  # At threshold → fully opaque
    otf.add_point(1.0, 1.0)  # Everything else remains fully visible

    return otf
	
def showvol(vol, vlim=None, colormap="jet", show_axes=True, show_colorbar=True,
			opacity_mode = 'adaptive', thr = 0.05):
    '''
    Volume rendering using Mayavi mlab with customization options.
    
    Parameters:
        vol (np.ndarray): 3D volume data.
        vlim (tuple): (vmin, vmax) for intensity scaling.
        colormap (str): Colormap to use (e.g., "jet", "viridis", "gray").
        show_axes (bool): Whether to display the coordinate axes.
        show_colorbar (bool): Whether to show a colorbar.
    '''
    if vlim is None:
        vmin = 0
        vmax = max(vol)
    else:
        vmin, vmax = vlim
    
    # Ensure the figure is managed by mlab
    fig = mlab.gcf()  # Get the current figure

    # Create volume rendering explicitly linked to the figure
    src = mlab.pipeline.scalar_field(vol, figure=fig)
    volume = mlab.pipeline.volume(src, vmin=vmin, vmax=vmax, figure=fig)
    # volume._volume_mapper.sample_distance = 0.01

    # Convert colormap to ColorTransferFunction and apply it
    ctf = cmap_to_ctf(colormap, vmin, vmax)
    volume._volume_property.set_color(ctf)
    volume._ctf = ctf
    volume.update_ctf = True

    # # Apply Adaptive Opacity Transfer Function
    if opacity_mode == 'binary':
        otf = create_binary_opacity_function(thr)
    elif opacity_mode == 'fade':
        otf = create_fade_opacity_function(vmin, vmax)
    elif opacity_mode == 'adaptive':
        otf = create_adaptive_opacity_function(vol, vmin, vmax)
    elif opacity_mode == 'solid':
        otf = create_solid_opacity_function()
		
    volume._volume_property.set_scalar_opacity(otf)
    volume._otf = otf
    volume.update_ctf = True 

    # Extract and apply the same LUT to the colorbar
    lut_manager = volume.module_manager.scalar_lut_manager
    lut_manager.lut_mode = colormap  # Ensure the colorbar follows the colormap

    # Show colorbar
    if show_colorbar:
        mlab.colorbar(orientation="vertical", title="Intensity")

    # Toggle axes visibility
    if show_axes:
        mlab.orientation_axes()
    else:
        mlab.axes(visible=False)


# Create a 3D array of zeros
shape = (256, 256, 256)  # (x, y, z)
vol = np.zeros(shape, dtype='float32')

# vol = create_sphere(vol, center=(100,100,100), radius=50, fill_value=1)
# vol = create_sphere_hollow(vol, center=(100,100,100), outer_radius=50, thickness = 5, fill_value=1)
vol = create_cylinder(vol, center_xy=(99, 99), z_range = (40, 161), radius=50, fill_value=1)
# vol = create_cylinder_hollow(vol, center_xy=(99, 99), z_range=(40, 161), 
#                              outer_radius=50, thickness=5, fill_value=1, caps=True)

# vol = create_cone(vol, tip=(50,50,50), height=75, base_radius=100, fill_value=1)
# vol = create_cone_hollow(vol, tip=(100,100,50), height=75, outer_radius=100, thickness = 10, 
                        #  fill_value=1, caps=True)
# vol = create_pyramid(vol, tip=(100,100,50), height=75, base_size=100, fill_value=1)
# vol = create_pyramid_hollow(vol, tip=(100,100,50), height=75, base_size=100, fill_value=1)

# vol = create_torus(vol, center=(100,100,100), major_radius=75, minor_radius=24, fill_value=1, hollow_thickness=5)
# vol = create_ellipsoid(vol, center=(100,100,100), radii=(50, 90, 75), fill_value=1, hollow_thickness=0)

# vol = create_hexagonal_prism(vol, center_xy=(100,100), z_range=(40,160), outer_radius=10, fill_value=1)
# vol = create_mobius_strip(vol, center=(100,100,100), major_radius=100, width=100, num_points=10000, fill_value=1)
# vol = create_menger_sponge(vol, center=(100,100,100), size=100, depth=3, fill_value=1)

# num_points = 250
# seed_points = [(np.random.randint(0, shape[0]), np.random.randint(0, shape[1]), np.random.randint(0, shape[2])) for _ in range(num_points)]

# # Generate 3D Voronoi diagram
# start = time.time()
# vol = create_voronoi_3d(vol, seed_points)
# print(f"Time taken: {time.time() - start:.2f} seconds")

optimal_distance = max(shape) * 3 
cx, cy, cz = np.array(shape) / 2  # ✅ Compute focal point
mlab.figure(bgcolor=(1, 1, 1))
showvol(vol, opacity_mode = 'binary', thr = 0.1)
mlab.view(elevation=90, distance=optimal_distance, focalpoint=(cx, cy, cz))
# mlab.close()
