# -*- coding: utf-8 -*-
"""
Methods for creating geometric shapes in 2D.

This module provides a collection of functions for generating and manipulating 
various 2D geometric shapes within NumPy arrays. The shapes include basic primitives 
such as circles, ellipses, rectangles, and triangles, as well as more complex 
structures like polygons, stars, and Voronoi diagrams.

Each function operates directly on a 2D NumPy array, modifying it by filling the 
specified shape with a given value. The methods utilize vectorized operations 
for efficiency and support both solid and hollow versions of the shapes.

These functions are useful for synthetic data generation, image processing, 
pattern recognition, and visualization tasks in scientific computing.

@author: Dr A. Vamvakeros
"""

import numpy as np

def create_circle(arr, center, radius, thickness=0, fill_value=1):
    """
    Creates a solid or hollow circle (ring) in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the circle will be drawn.
    center : tuple of int
        The (x, y) coordinates of the circle's center.
    radius : int
        The outer radius of the circle.
    thickness : int, optional (default=0)
        Thickness of the ring. If `thickness=0`, the circle is solid.
        Otherwise, a hollow ring is created with an inner radius of `radius - thickness`.
    fill_value : int or float, optional (default=1)
        The value to fill inside the circle or ring.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn circle or ring.
    """
    cx, cy = center
    inner_radius = max(radius - thickness, 0)

    H, W = arr.shape
    X, Y = np.ogrid[:H, :W]

    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    outer_r2 = radius ** 2
    inner_r2 = inner_radius ** 2

    mask = dist2 <= outer_r2 if thickness == 0 else (dist2 <= outer_r2) & (dist2 >= inner_r2)
    arr[mask] = fill_value
    return arr

def create_rectangle(arr, corner, width, height, thickness=0, fill_value=1):
    """
    Creates a solid or hollow rectangle in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the rectangle will be drawn.
    corner : tuple of int
        The (x, y) coordinates of the top-left corner of the rectangle.
    width : int
        The width of the rectangle (along the y-axis).
    height : int
        The height of the rectangle (along the x-axis).
    thickness : int, optional (default=0)
        Thickness of the border. If `thickness=0`, the rectangle is solid.
        Otherwise, a hollow rectangle is created with an inner area left unfilled.
    fill_value : int or float, optional (default=1)
        The value to fill inside the rectangle or its border.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn rectangle.
    """
    x0, y0 = corner
    H, W = arr.shape

    # Bound the rectangle so it does not exceed the array limits
    x_end = min(x0 + height, H)
    y_end = min(y0 + width, W)

    if thickness == 0:
        # Solid rectangle
        arr[x0:x_end, y0:y_end] = fill_value
    else:
        # Hollow rectangle (border only)
        arr[x0:x0+thickness, y0:y_end] = fill_value  # Top edge
        arr[x_end-thickness:x_end, y0:y_end] = fill_value  # Bottom edge
        arr[x0:x_end, y0:y0+thickness] = fill_value  # Left edge
        arr[x0:x_end, y_end-thickness:y_end] = fill_value  # Right edge

    return arr

def create_triangle(arr, p1, p2, p3, fill_value=1):
    """
    Creates a solid or hollow triangle in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the triangle will be drawn.
    p1 : tuple of int
        Coordinates (x, y) of the first vertex of the triangle.
    p2 : tuple of int
        Coordinates (x, y) of the second vertex of the triangle.
    p3 : tuple of int
        Coordinates (x, y) of the third vertex of the triangle.
    fill_value : int or float, optional (default=1)
        The value to fill inside the triangle or its border.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn triangle.
    """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    H, W = arr.shape

    # Compute bounding box of the triangle
    min_x = max(min(x1, x2, x3), 0)
    max_x = min(max(x1, x2, x3), H - 1)
    min_y = max(min(y1, y2, y3), 0)
    max_y = min(max(y1, y2, y3), W - 1)

    if min_x > max_x or min_y > max_y:
        return arr  # No area to fill

    X, Y = np.ogrid[min_x:max_x+1, min_y:max_y+1]

    # Compute triangle area using determinant method
    def tri_area(ax, ay, bx, by, cx, cy):
        return np.abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    area_ABC = tri_area(x1, y1, x2, y2, x3, y3)
    area_PBC = tri_area(X, Y, x2, y2, x3, y3)
    area_PAC = tri_area(x1, y1, X, Y, x3, y3)
    area_PAB = tri_area(x1, y1, x2, y2, X, Y)


    inside_mask = (area_PBC + area_PAC + area_PAB) == area_ABC
    arr_region = arr[min_x:max_x+1, min_y:max_y+1]
    arr_region[inside_mask] = fill_value

    return arr


def create_equilateral_triangle(arr, p1, side, fill_value=1, orientation='default'):
    """
    Creates and fills an equilateral triangle in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the triangle will be drawn.
    p1 : tuple of int
        Coordinates (x, y) for the first vertex (base-left) of the triangle.
    side : int or float
        The length of each side of the equilateral triangle.
    fill_value : int or float, optional (default=1)
        The value used to fill the triangle in the array.
    orientation : str, optional (default='default')
        Specifies the orientation of the triangle:
        - 'default': Base is horizontal, apex above the base.
        - 'down': Base is horizontal, apex below the base.
        - 'random': Base is at a random angle.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn triangle.
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

def create_ellipse(arr, center, axes_radii, thickness=0, fill_value=1):
    """
    Creates a solid or hollow ellipse in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the ellipse will be drawn.
    center : tuple of float
        Coordinates (cx, cy) for the center of the ellipse (row, col).
    axes_radii : tuple of float
        (a, b): The radii of the semi-major and semi-minor axes.
    thickness : int, optional (default=0)
        Thickness of the hollow ring. If set to 0, the ellipse is solid.
    fill_value : int or float, optional (default=1)
        The value used to fill the ellipse in the array.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn ellipse.
    """
    cx, cy = center
    a_outer, b_outer = axes_radii

    H, W = arr.shape
    X, Y = np.ogrid[:H, :W]

    # Outer ellipse mask
    outer_mask = ((X - cx)**2 / a_outer**2) + ((Y - cy)**2 / b_outer**2) <= 1

    if thickness > 0:
        # Ensure valid inner ellipse radii
        a_inner = max(a_outer - thickness, 0.1)
        b_inner = max(b_outer - thickness, 0.1)
        
        # Inner ellipse mask
        inner_mask = ((X - cx)**2 / a_inner**2) + ((Y - cy)**2 / b_inner**2) <= 1
        
        # Hollow ring mask
        final_mask = outer_mask & (~inner_mask)
    else:
        final_mask = outer_mask

    arr[final_mask] = fill_value
    return arr

def create_star(arr, center, n_points=5, r_outer=10, r_inner=5, fill_value=1, angle_offset=0.0):
    """
    Creates a filled star shape in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the star will be drawn.
    center : tuple of int
        Coordinates (cx, cy) for the center of the star (row, col).
    n_points : int, optional (default=5)
        Number of points in the star.
    r_outer : float, optional (default=10)
        Radius of the outer points of the star.
    r_inner : float, optional (default=5)
        Radius of the inner points of the star.
    fill_value : int or float, optional (default=1)
        The value used to fill the star in the array.
    angle_offset : float, optional (default=0.0)
        Angle offset in radians to rotate the star.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn star.
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
    Creates and fills an arbitrary polygon in a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D array where the polygon will be drawn.
    vertices : list of tuples
        List of (x, y) coordinates representing the polygon vertices.
    fill_value : int or float, optional (default=1)
        The value used to fill the polygon in the array.

    Returns
    -------
    numpy.ndarray
        The modified array with the drawn polygon.
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
    Determines if given points are inside a polygon using the ray-casting method.

    Parameters
    ----------
    xs : numpy.ndarray
        1D array of x-coordinates of points to test.
    ys : numpy.ndarray
        1D array of y-coordinates of points to test.
    vertices : list of tuples
        List of (x, y) coordinates representing the polygon vertices.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating whether each point is inside the polygon.
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
    Generates a Voronoi diagram in a 2D NumPy array based on given seed points.

    Parameters
    ----------
    arr : np.ndarray
        A 2D array where the Voronoi diagram will be drawn.
    seed_points : list of tuples
        A list of coordinates (x, y) representing the Voronoi seed points.
    fill_values : list or np.ndarray, optional
        An array or list of values assigned to each region. If None, assigns unique
        integer values to each Voronoi cell.

    Returns
    -------
    np.ndarray
        The modified array with Voronoi regions assigned.
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
