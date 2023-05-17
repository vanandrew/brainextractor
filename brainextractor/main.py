"""
    Main BrainExtractor class
"""
import os
import warnings
import numpy as np
import nibabel as nib
import trimesh
from numba import jit
from numba.typed import List
from .helpers import sphere, closest_integer_point, bresenham3d, l2norm, l2normarray, diagonal_dot


class BrainExtractor:
    """
    Implemenation of the FSL Brain Extraction Tool

    This class takes in a Nifti1Image class and generates
    the brain surface and mask.
    """

    def __init__(
        self,
        img: nib.Nifti1Image,
        t02t: float = 0.02,
        t98t: float = 0.98,
        bt: float = 0.5,
        d1: float = 20.0,  # mm
        d2: float = 10.0,  # mm
        rmin: float = 3.33,  # mm
        rmax: float = 10.0,  # mm
    ):
        """
        Initialization of Brain Extractor

        Computes image range/thresholds and
        estimates the brain radius
        """
        print("Initializing...")

        # get image resolution
        res = img.header["pixdim"][1]
        if not np.allclose(res, img.header["pixdim"][1:4], rtol=1e-3):
            warnings.warn(
                "The voxels in this image are non-isotropic! \
                Brain extraction settings may not be valid!"
            )

        # store brain extraction parameters
        print("Parameters: bt=%f, d1=%f, d2=%f, rmin=%f, rmax=%f" % (bt, d1, d2, rmin, rmax))
        self.bt = bt
        self.d1 = d1 / res
        self.d2 = d2 / res
        self.rmin = rmin / res
        self.rmax = rmax / res

        # compute E, F constants
        self.E = (1.0 / rmin + 1.0 / rmax) / 2.0
        self.F = 6.0 / (1.0 / rmin - 1.0 / rmax)

        # store the image
        self.img = img

        # store conveinent references
        self.data = img.get_fdata()  # 3D data
        self.rdata = img.get_fdata().ravel()  # flattened data
        self.shape = img.shape  # 3D shape
        self.rshape = np.multiply.reduce(img.shape)  # flattened shape

        # get thresholds from histogram
        sorted_data = np.sort(self.rdata)
        self.tmin = np.min(sorted_data)
        self.t2 = sorted_data[np.ceil(t02t * self.rshape).astype(np.int64) + 1]
        self.t98 = sorted_data[np.ceil(t98t * self.rshape).astype(np.int64) + 1]
        self.tmax = np.max(sorted_data)
        self.t = (self.t98 - self.t2) * 0.1 + self.t2
        print("tmin: %f, t2: %f, t: %f, t98: %f, tmax: %f" % (self.tmin, self.t2, self.t, self.t98, self.tmax))

        # find the center of mass of image
        ic, jc, kc = np.meshgrid(
            np.arange(self.shape[0]), np.arange(self.shape[1]), np.arange(self.shape[2]), indexing="ij", copy=False
        )
        cdata = np.clip(self.rdata, self.t2, self.t98) * (self.rdata > self.t)
        ci = np.average(ic.ravel(), weights=cdata)
        cj = np.average(jc.ravel(), weights=cdata)
        ck = np.average(kc.ravel(), weights=cdata)
        self.c = np.array([ci, cj, ck])
        print("Center-of-Mass: {}".format(self.c))

        # compute 1/2 head radius with spherical formula
        self.r = 0.5 * np.cbrt(3 * np.sum(self.rdata > self.t) / (4 * np.pi))
        print("Head Radius: %f" % (2 * self.r))

        # get median value within estimated head sphere
        self.tm = np.median(self.data[sphere(self.shape, 2 * self.r, self.c)])
        print("Median within Head Radius: %f" % self.tm)

        # generate initial surface
        print("Initializing surface...")
        self.surface = trimesh.creation.icosphere(subdivisions=4, radius=self.r)
        self.surface = self.surface.apply_transform([[1, 0, 0, ci], [0, 1, 0, cj], [0, 0, 1, ck], [0, 0, 0, 1]])

        # update the surface attributes
        self.num_vertices = self.surface.vertices.shape[0]
        self.num_faces = self.surface.faces.shape[0]
        self.vertices = np.array(self.surface.vertices)
        self.faces = np.array(self.surface.faces)
        self.vertex_neighbors_idx = List([np.array(i) for i in self.surface.vertex_neighbors])
        # compute location of vertices in face array
        self.face_vertex_idxs = np.zeros((self.num_vertices, 6, 2), dtype=np.int64)
        for v in range(self.num_vertices):
            f, i = np.asarray(self.faces == v).nonzero()
            self.face_vertex_idxs[v, : i.shape[0], 0] = f
            self.face_vertex_idxs[v, : i.shape[0], 1] = i
            if i.shape[0] == 5:
                self.face_vertex_idxs[v, 5, 0] = -1
                self.face_vertex_idxs[v, 5, 1] = -1
        self.update_surface_attributes()
        print("Brain extractor initialization complete!")

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_face_normals(num_faces, faces, vertices):
        """
        Compute face normals
        """
        face_normals = np.zeros((num_faces, 3))
        for i, f in enumerate(faces):
            local_v = vertices[f]
            a = local_v[1] - local_v[0]
            b = local_v[2] - local_v[0]
            face_normals[i] = np.array(
                (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            )
            face_normals[i] /= l2norm(face_normals[i])
        return face_normals

    @staticmethod
    def compute_face_angles(triangles: np.ndarray):
        """
        Compute angles in triangles of each face
        """
        # don't copy triangles
        triangles = np.asanyarray(triangles, dtype=np.float64)

        # get a unit vector for each edge of the triangle
        u = triangles[:, 1] - triangles[:, 0]
        u /= l2normarray(u)[:, np.newaxis]
        v = triangles[:, 2] - triangles[:, 0]
        v /= l2normarray(v)[:, np.newaxis]
        w = triangles[:, 2] - triangles[:, 1]
        w /= l2normarray(w)[:, np.newaxis]

        # run the cosine and per-row dot product
        result = np.zeros((len(triangles), 3), dtype=np.float64)
        # clip to make sure we don't float error past 1.0
        result[:, 0] = np.arccos(np.clip(diagonal_dot(u, v), -1, 1))
        result[:, 1] = np.arccos(np.clip(diagonal_dot(-u, w), -1, 1))
        # the third angle is just the remaining
        result[:, 2] = np.pi - result[:, 0] - result[:, 1]

        # a triangle with any zero angles is degenerate
        # so set all of the angles to zero in that case
        result[(result < 1e-8).any(axis=1), :] = 0.0
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_vertex_normals(
        num_vertices: int,
        faces: np.ndarray,
        face_normals: np.ndarray,
        face_angles: np.ndarray,
        face_vertex_idxs: np.ndarray,
    ):
        """
        Computes vertex normals

        Sums face normals connected to vertex, weighting
        by the angle the vertex makes with the face
        """
        vertex_normals = np.zeros((num_vertices, 3))
        for vertex_idx in range(num_vertices):
            face_idxs = np.asarray([f for f in face_vertex_idxs[vertex_idx, :, 0] if f != -1])
            inface_idxs = np.asarray([f for f in face_vertex_idxs[vertex_idx, :, 1] if f != -1])
            surrounding_angles = face_angles.ravel()[face_idxs * 3 + inface_idxs]
            vertex_normals[vertex_idx] = np.dot(surrounding_angles / surrounding_angles.sum(), face_normals[face_idxs])
            vertex_normals[vertex_idx] /= l2norm(vertex_normals[vertex_idx])
        return vertex_normals

    def rebuild_surface(self):
        """
        Rebuilds the surface mesh for given updated vertices
        """
        self.update_surface_attributes()
        self.surface = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    @staticmethod
    @jit(nopython=True, cache=True)
    def update_surf_attr(vertices: np.ndarray, neighbors_idx: list):
        # the neighbors array is tricky because it doesn't
        # have the structure of a nice rectangular array
        # we initialize it to be the largest size (6) then we
        # can make a check for valid vertices later with neighbors size
        neighbors = np.zeros((vertices.shape[0], 6, 3))
        neighbors_size = np.zeros(vertices.shape[0], dtype=np.int8)
        for i, ni in enumerate(neighbors_idx):
            for j, vi in enumerate(ni):
                neighbors[i, j, :] = vertices[vi]
            neighbors_size[i] = j + 1

        # compute centroids
        centroids = np.zeros((vertices.shape[0], 3))
        for i, (n, s) in enumerate(zip(neighbors, neighbors_size)):
            centroids[i, 0] = np.mean(n[:s, 0])
            centroids[i, 1] = np.mean(n[:s, 1])
            centroids[i, 2] = np.mean(n[:s, 2])

        # return optimized surface attributes
        return neighbors, neighbors_size, centroids

    def update_surface_attributes(self):
        """
        Updates attributes related to the surface
        """
        self.triangles = self.vertices[self.faces]
        self.face_normals = self.compute_face_normals(self.num_faces, self.faces, self.vertices)
        self.face_angles = self.compute_face_angles(self.triangles)
        self.vertex_normals = self.compute_vertex_normals(
            self.num_vertices, self.faces, self.face_normals, self.face_angles, self.face_vertex_idxs
        )
        self.vertex_neighbors, self.vertex_neighbors_size, self.vertex_neighbors_centroids = self.update_surf_attr(
            self.vertices, self.vertex_neighbors_idx
        )
        self.l = self.get_mean_intervertex_distance(self.vertices, self.vertex_neighbors, self.vertex_neighbors_size)

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_mean_intervertex_distance(vertices: np.ndarray, neighbors: np.ndarray, sizes: np.ndarray):
        """
        Computes the mean intervertex distance across the entire surface
        """
        mivd = np.zeros(vertices.shape[0])
        for v in range(vertices.shape[0]):
            vecs = vertices[v] - neighbors[v, : sizes[v]]
            vd = np.zeros(vecs.shape[0])
            for i in range(vecs.shape[0]):
                vd[i] = l2norm(vecs[i])
            mivd[v] = np.mean(vd)
        return np.mean(mivd)

    def run(self, iterations: int = 1000, deformation_path: str = None):
        """
        Runs the extraction step.

        This deforms the surface based on the method outlined in"

        Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
        2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568;
        PMCID: PMC6871816.

        """
        print("Running surface deformation...")
        # initialize s_vectors
        s_vectors = np.zeros(self.vertices.shape)

        # initialize s_vector normal/tangent
        s_n = np.zeros(self.vertices.shape)
        s_t = np.zeros(self.vertices.shape)

        # initialize u components
        u1 = np.zeros(self.vertices.shape)
        u2 = np.zeros(self.vertices.shape)
        u3 = np.zeros(self.vertices.shape)
        u = np.zeros(self.vertices.shape)

        # if deformation path defined
        if deformation_path:
            import zipfile

            zip_file = zipfile.ZipFile(deformation_path, "w")

        # surface deformation loop
        for i in range(iterations):
            print("Iteration: %d" % i, end="\r")
            # run one step of deformation
            self.step_of_deformation(
                self.data,
                self.vertices,
                self.vertex_normals,
                self.vertex_neighbors_centroids,
                self.l,
                self.t2,
                self.t,
                self.tm,
                self.t98,
                self.E,
                self.F,
                self.bt,
                self.d1,
                self.d2,
                s_vectors,
                s_n,
                s_t,
                u1,
                u2,
                u3,
                u,
            )
            # update vertices
            self.vertices += u
            if deformation_path:  # write to stl if enabled
                surface_file = "surface{:0>5d}.stl".format(i)
                dirpath = os.path.dirname(deformation_path)
                self.rebuild_surface()
                self.save_surface(os.path.join(dirpath, surface_file))
                zip_file.write(os.path.join(dirpath, surface_file), surface_file)
                os.remove(os.path.join(dirpath, surface_file))
            else:  # just update the surface attributes
                self.update_surface_attributes()

        # close zip file
        if deformation_path:
            zip_file.close()

        # update the surface
        self.rebuild_surface()
        print("")
        print("Complete.")

    @staticmethod
    @jit(nopython=True, cache=True)
    def step_of_deformation(
        data: np.ndarray,
        vertices: np.ndarray,
        normals: np.ndarray,
        neighbors_centroids: np.ndarray,
        l: float,
        t2: float,
        t: float,
        tm: float,
        t98: float,
        E: float,
        F: float,
        bt: float,
        d1: float,
        d2: float,
        s_vectors: np.ndarray,
        s_n: np.ndarray,
        s_t: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray,
        u3: np.ndarray,
        u: np.ndarray,
    ):
        """
        Finds a single displacement step for the surface
        """
        # loop over vertices
        for i, vertex in enumerate(vertices):
            # compute s vector
            s_vectors[i] = neighbors_centroids[i] - vertex

            # split s vector into normal and tangent components
            s_n[i] = np.dot(s_vectors[i], normals[i]) * normals[i]
            s_t[i] = s_vectors[i] - s_n[i]

            # set component u1
            u1[i] = 0.5 * s_t[i]

            # compute local radius of curvature
            r = (l ** 2) / (2 * l2norm(s_n[i]))

            # compute f2
            f2 = (1 + np.tanh(F * (1 / r - E))) / 2

            # set component u2
            u2[i] = f2 * s_n[i]

            # get endpoints directed interior (distance set by d1 and d2)
            e1 = closest_integer_point(vertex - d1 * normals[i])
            e2 = closest_integer_point(vertex - d2 * normals[i])

            # get lines created by e1/e2
            c = closest_integer_point(vertex)
            i1 = bresenham3d(c, e1)
            i2 = bresenham3d(c, e2)

            # get Imin/Imax
            linedata1 = [data[d[0], d[1], d[2]] for d in i1]
            linedata1.append(tm)
            linedata1 = np.asarray(linedata1)
            Imin = np.max(np.asarray([t2, np.min(linedata1)]))
            linedata2 = [data[d[0], d[1], d[2]] for d in i2]
            linedata2.append(t)
            linedata2 = np.asarray(linedata2)
            Imax = np.min(np.asarray([tm, np.max(linedata2)]))

            # get tl
            tl = (Imax - t2) * bt + t2

            # compute f3
            f3 = 0.05 * 2 * (Imin - tl) / (Imax - t2) * l

            # get component u3
            u3[i] = f3 * normals[i]

        # get displacement vector
        u[:, :] = u1 + u2 + u3

    @staticmethod
    def check_bound(img_min: int, img_max: int, img_start: int, img_end: int, vol_start: int, vol_end: int):
        if img_min < img_start:
            vol_start = vol_start + (img_start - img_min)
            img_min = 0
        if img_max > img_end:
            vol_end = vol_end - (img_max - img_end)
            img_max = img_end
        return img_min, img_max, img_start, img_end, vol_start, vol_end

    def compute_mask(self):
        """
        Convert surface mesh to volume
        """
        vol = self.surface.voxelized(1)
        vol = vol.fill()
        self.mask = np.zeros(self.shape)
        bounds = vol.bounds

        # adjust bounds to handle data outside the field of view

        # get the bounds of the volumized surface mesh
        x_min = int(vol.bounds[0, 0]) if vol.bounds[0, 0] > 0 else int(vol.bounds[0, 0]) - 1
        x_max = int(vol.bounds[1, 0]) if vol.bounds[1, 0] > 0 else int(vol.bounds[1, 0]) - 1
        y_min = int(vol.bounds[0, 1]) if vol.bounds[0, 1] > 0 else int(vol.bounds[0, 1]) - 1
        y_max = int(vol.bounds[1, 1]) if vol.bounds[1, 1] > 0 else int(vol.bounds[1, 1]) - 1
        z_min = int(vol.bounds[0, 2]) if vol.bounds[0, 2] > 0 else int(vol.bounds[0, 2]) - 1
        z_max = int(vol.bounds[1, 2]) if vol.bounds[1, 2] > 0 else int(vol.bounds[1, 2]) - 1

        # get the extents of the original image
        x_start = 0
        y_start = 0
        z_start = 0
        x_end = int(self.shape[0])
        y_end = int(self.shape[1])
        z_end = int(self.shape[2])

        # get the extents of the volumized surface
        x_vol_start = 0
        y_vol_start = 0
        z_vol_start = 0
        x_vol_end = int(vol.matrix.shape[0])
        y_vol_end = int(vol.matrix.shape[1])
        z_vol_end = int(vol.matrix.shape[2])

        # if the volumized surface mesh is outside the extents of the original image
        # we need to crop this volume to fit the image
        x_min, x_max, x_start, x_end, x_vol_start, x_vol_end = self.check_bound(
            x_min, x_max, x_start, x_end, x_vol_start, x_vol_end
        )
        y_min, y_max, y_start, y_end, y_vol_start, y_vol_end = self.check_bound(
            y_min, y_max, y_start, y_end, y_vol_start, y_vol_end
        )
        z_min, z_max, z_start, z_end, z_vol_start, z_vol_end = self.check_bound(
            z_min, z_max, z_start, z_end, z_vol_start, z_vol_end
        )
        self.mask[x_min:x_max, y_min:y_max, z_min:z_max] = vol.matrix[
            x_vol_start:x_vol_end, y_vol_start:y_vol_end, z_vol_start:z_vol_end
        ]
        return self.mask

    def save_mask(self, filename: str):
        """
        Saves brain extraction to nifti file
        """
        mask = self.compute_mask()
        nib.Nifti1Image(mask, self.img.affine).to_filename(filename)

    def save_surface(self, filename: str):
        """
        Save surface in .stl
        """
        self.surface.export(filename)
