# -*- coding: utf-8 -*-
"""
Methods for creating shapes in 2D and 3D

@author: Antony Vamvakeros
"""

#%%

import numpy as np


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


def fill_hollow_circle_vectorized(arr, center, outer_radius, thickness=1, fill_value=1):
    """
    Fills a hollow circle (ring) in 2D array `arr`.
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


def fill_rectangle_corner_vectorized(arr, corner, width, height, fill_value=1):
    """
    Fills a rectangle given by its top-left corner (x0, y0) 
    and dimensions `width` (along y) and `height` (along x),
    using slicing (vectorized).
    """
    x0, y0 = corner
    H, W = arr.shape
    
    # Bound the rectangle so we don't go out of array
    x_end = min(x0 + height, H)
    y_end = min(y0 + width, W)
    
    arr[x0:x_end, y0:y_end] = fill_value


def fill_hollow_rectangle_corner_vectorized(arr, corner, width, height, thickness=1, fill_value=1):
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


def fill_triangle_vectorized(arr, p1, p2, p3, fill_value=1):
    """
    Fills a triangle in 2D array `arr` using a vectorized
    approach with a bounding-box mask and a barycentric check.
    This is more advanced than the simple loop approach.
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


def fill_hollow_triangle_vectorized(arr, p1, p2, p3, thickness=1, fill_value=1):
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


def draw_line_thick_vectorized(arr, p1, p2, thickness=1, fill_value=1):
    """
    Draw a 'thick' line in a 2D array `arr` by computing the 
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


#%%

import matplotlib.pyplot as plt
%matplotlib qt

#%%

im = np.zeros((256, 256), dtype='float32')

imn = create_circle(im, center=(50,50), radius=10, fill_value=2)
imn = create_circle(imn, center=(25,25), radius=20, fill_value=1)

plt.figure(1);plt.clf()
plt.imshow(imn, cmap='gray', interpolation='None')
plt.colorbar()
plt.show()





#%%

# Create a 3D array of zeros
shape = (50, 50, 50)  # (x, y, z)
volume = np.zeros(shape, dtype=int)

# 1) Solid sphere
fill_sphere(volume, center=(25, 25, 25), radius=8, fill_value=1)

# 2) Hollow sphere
fill_hollow_sphere(volume, center=(25, 25, 25), outer_radius=15, thickness=2, fill_value=2)

# 3) Solid cube (corner-based)
fill_cube_corner(volume, corner=(0, 0, 0), size=5, fill_value=3)

# 4) Hollow cube (corner-based)
fill_hollow_cube_corner(volume, corner=(10, 10, 10), size=8, wall_thickness=1, fill_value=4)

# 5) Solid cylinder (z-axis)
fill_cylinder(volume, center_xy=(25, 25), z_range=(0, 20), radius=5, fill_value=5)

# 6) Hollow cylinder
fill_hollow_cylinder(volume, center_xy=(35, 35), z_range=(0, 30), outer_radius=5, thickness=1, fill_value=6)

# 7) Solid cone
fill_cone(volume, tip=(5, 5, 0), height=10, base_radius=5, fill_value=7)

# 8) Hollow cone
fill_hollow_cone(volume, tip=(15, 15, 0), height=15, outer_radius=7, thickness=2, fill_value=8)
