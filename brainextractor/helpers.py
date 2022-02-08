"""
    Helper functions
"""
import numpy as np
import trimesh
from numba import jit


def sphere(shape: list, radius: float, position: list):
    """
    Creates a binary sphere
    """
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


@jit(nopython=True, cache=True)
def closest_integer_point(vertex: np.ndarray):
    """
    Gives the closest integer point based on euclidean distance
    """
    # get neighboring grid points to search
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    x0 = np.floor(x)
    y0 = np.floor(y)
    z0 = np.floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # initialize min euclidean distance
    min_euclid = 99

    # loop through each neighbor point
    for i in [x0, x1]:
        for j in [y0, y1]:
            for k in [z0, z1]:
                # compare coordinate and store if min euclid distance
                coords = np.array([i, j, k])
                dist = l2norm(vertex - coords)
                if dist < min_euclid:
                    min_euclid = dist
                    final_coords = coords

    # return the final coords
    return final_coords.astype(np.int64)


@jit(nopython=True, cache=True)
def bresenham3d(v0: np.ndarray, v1: np.ndarray):
    """
    Bresenham's algorithm for a 3-D line

    https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
    """
    # initialize axis differences

    dx = np.abs(v1[0] - v0[0])
    dy = np.abs(v1[1] - v0[1])
    dz = np.abs(v1[2] - v0[2])
    xs = 1 if (v1[0] > v0[0]) else -1
    ys = 1 if (v1[1] > v0[1]) else -1
    zs = 1 if (v1[2] > v0[2]) else -1

    # determine the driving axis
    if dx >= dy and dx >= dz:
        d0 = dx
        d1 = dy
        d2 = dz
        s0 = xs
        s1 = ys
        s2 = zs
        a0 = 0
        a1 = 1
        a2 = 2
    elif dy >= dx and dy >= dz:
        d0 = dy
        d1 = dx
        d2 = dz
        s0 = ys
        s1 = xs
        s2 = zs
        a0 = 1
        a1 = 0
        a2 = 2
    elif dz >= dx and dz >= dy:
        d0 = dz
        d1 = dx
        d2 = dy
        s0 = zs
        s1 = xs
        s2 = ys
        a0 = 2
        a1 = 0
        a2 = 1

    # create line array
    line = np.zeros((d0 + 1, 3), dtype=np.int64)
    line[0] = v0

    # get points
    p1 = 2 * d1 - d0
    p2 = 2 * d2 - d0
    for i in range(d0):
        c = line[i].copy()
        c[a0] += s0
        if p1 >= 0:
            c[a1] += s1
            p1 -= 2 * d0
        if p2 >= 0:
            c[a2] += s2
            p2 -= 2 * d0
        p1 += 2 * d1
        p2 += 2 * d2
        line[i + 1] = c

    # return list
    return line


@jit(nopython=True, cache=True)
def l2norm(vec: np.ndarray):
    """
    Computes the l2 norm for 3d vector
    """
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@jit(nopython=True, cache=True)
def l2normarray(array: np.ndarray):
    """
    Computes the l2 norm for several 3d vectors
    """
    return np.sqrt(array[:, 0] ** 2 + array[:, 1] ** 2 + array[:, 2] ** 2)


def diagonal_dot(a: np.ndarray, b: np.ndarray):
    """
    Dot product by row of a and b.
    There are a lot of ways to do this though
    performance varies very widely. This method
    uses a dot product to sum the row and avoids
    function calls if at all possible.
    """
    a = np.asanyarray(a)
    return np.dot(a * b, [1.0] * a.shape[1])
