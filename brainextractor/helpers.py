"""
    Helper functions
"""
import numpy as np
import trimesh
from numba import jit
from scipy.spatial import cKDTree # pylint: disable=no-name-in-module

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

def cartesian(arrays, out=None):
    """
        Generate a cartesian product of input arrays
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def find_enclosure(surface: trimesh.Trimesh, data_shape: tuple):
    """
        Finds all voxels inside of a surface

        This function stores all the surface vertices in a k-d tree
        and uses it to quickly look up the closest vertex to each
        volume voxel in the image.

        Once the closest vertex is found, a vector is created between
        the voxel location and the vertex. The resulting vector is dot
        product with the corresponding vertex normal. Negative values
        indicate that the voxel lies exterior to the surface (since it
        is anti-parallel to the vertex normal), while positive values
        indicate that they are interior to the surface (parallel to
        the vertex normal).
    """
    # get vertex normals for each vertex on the surface
    normals = surface.vertex_normals

    # create KDTree over surface vertices
    searcher = cKDTree(surface.vertices)

    # get bounding box around surface
    max_loc = np.ceil(np.max(surface.vertices, axis=0)).astype(np.int64)
    min_loc = np.floor(np.min(surface.vertices, axis=0)).astype(np.int64)

    # build a list of locations representing the volume grid
    # within the bounding box
    locs = cartesian([
        np.arange(min_loc[0], max_loc[0]),
        np.arange(min_loc[1], max_loc[1]),
        np.arange(min_loc[2], max_loc[2])])

    # find the nearest vertex to each voxel
    # searcher.query returns a list of vertices corresponding
    # to the closest vertex to the given voxel location
    _, nearest_idx = searcher.query(locs, n_jobs=6)
    nearest_vertices = surface.vertices[nearest_idx]

    # get the directional vector from each voxel location to it's nearest vertex
    direction_vectors = nearest_vertices - locs

    # find it's direction by taking the dot product with vertex normal
    # this is done row-by-row between directional vectors and the vertex normals
    dot_products = np.einsum('ij,ij->i', direction_vectors, normals[nearest_idx])

    # get the interior (where dot product is > 0)
    interior = (dot_products > 0).reshape((max_loc - min_loc).astype(np.int64))

    # create mask
    mask = np.zeros(data_shape)
    mask[min_loc[0]:max_loc[0],min_loc[1]:max_loc[1],min_loc[2]:max_loc[2]] = interior

    # return the mask
    return mask

@jit(nopython=True, cache=True)
def closest_integer_point(vertex: np.ndarray):
    """
        Gives the closest integer point based on euclidena distance
    """
    # get neighboring grid points to search
    x = vertex[0]; y = vertex[1]; z = vertex[2]
    x0 = np.floor(x); y0 = np.floor(y); z0 = np.floor(z)
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1

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
        Bresenham's algorithm for 3-D line
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
        d0 = dx; d1 = dy; d2 = dz
        s0 = xs; s1 = ys; s2 = zs
        a0 = 0; a1 = 1; a2 = 2
    elif dy >= dx and dy >= dz:
        d0 = dy; d1 = dx; d2 = dz
        s0 = ys; s1 = xs; s2 = zs
        a0 = 1; a1 = 0; a2 = 2
    elif dz >= dx and dz >= dy:
        d0 = dz; d1 = dx; d2 = dy
        s0 = zs; s1 = xs; s2 = ys
        a0 = 2; a1 = 0; a2 = 1

    # create line array
    line = np.zeros((d0 + 1, 3), dtype=np.int64)
    line[0] = v0

    # get points
    p1 = 2*d1 - d0
    p2 = 2*d2 - d0
    for i in range(d0):
        c = line[i].copy()
        c[a0] += s0
        if (p1 >= 0):
            c[a1] += s1
            p1 -= 2 * d0
        if (p2 >= 0):
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
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)