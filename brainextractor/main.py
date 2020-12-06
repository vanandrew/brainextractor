"""
    Main BrainExtractor class
"""
import nibabel as nib
import numpy as np
import trimesh
from numba import jit, prange
from numba.typed import List
from .helpers import sphere, find_enclosure, closest_integer_point, bresenham3d, l2norm

class BrainExtractor:
    """
        Implemenation of the FSL Brain Extraction Tool

        This class takes in a Nifti1Image class and generates
        the brain surface and mask.
    """
    def __init__(self,
        img=nib.Nifti1Image,
        bt=0.1,
        d1=7.0, # mm
        d2=3.0, # mm
        rmin=3.33, # mm
        rmax=10.0 # mm
        ):
        """
            Initialization of Brain Extractor

            Computes image range/thresholds and 
            estimates the brain radius 
        """
        print("Initializing...")

        # store brain extraction parameters
        self.bt = bt**0.275 # FSL changes the fractional threshold to be bt^0.275 (Don't ask me why it does this...)
        self.d1 = d1
        self.d2 = d2
        self.rmin = rmin
        self.rmax = rmax

        # compute E, F constants
        self.E = (1.0/rmin + 1.0/rmax)/2.0
        self.F = 6.0/(1.0/rmin - 1.0/rmax)

        # store the image
        self.img = img

        # store conveinent references
        self.data = img.get_fdata() # 3D data
        self.rdata = img.get_fdata().ravel() # flattened data
        self.shape = img.shape # 3D shape
        self.rshape = np.multiply.reduce(img.shape) # flattened shape

        # get thresholds from histogram
        sorted_data = np.sort(self.rdata)
        self.tmin = np.min(sorted_data)
        self.t2 = sorted_data[np.ceil(0.02 * self.rshape).astype(np.int64) + 1]
        self.t98 = sorted_data[np.ceil(0.98 * self.rshape).astype(np.int64) + 1]
        self.tmax = np.max(sorted_data)
        self.t = (self.t98 - self.t2)*0.1 + self.t2
        print("tmin: %f, t2: %f, t: %f, t98: %f, tmax: %f" % (self.tmin, self.t2, self.t, self.t98, self.tmax))

        # find the center of mass of image
        ic, jc, kc = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), np.arange(self.shape[2]), indexing='ij', copy=False)
        cdata = np.clip(self.rdata, self.t2, self.t98) * (self.rdata > self.t)
        ci = np.average(ic.ravel(), weights=cdata)
        cj = np.average(jc.ravel(), weights=cdata)
        ck = np.average(kc.ravel(), weights=cdata)
        self.c = np.array([ci, cj, ck])
        print("Center-of-Mass: {}".format(self.c))

        # compute 1/2 head radius with spherical formula
        self.r = 0.5 * np.cbrt(3 * np.sum(self.rdata > self.t) / (4 * np.pi))
        print("Head Radius: %f" % (2*self.r))

        # get median value within estimated head sphere
        self.tm = np.median(self.data[sphere(self.shape, 2*self.r, self.c)])
        print("Median within Head Radius: %f" % self.tm)

        # generate initial surface
        print("Initializing surface...")
        self.surface = trimesh.creation.icosphere(subdivisions=4, radius=self.r)
        self.surface = self.surface.apply_transform([[1,0,0,ci],[0,1,0,cj],[0,0,1,ck],[0,0,0,1]])
        
        # update the surface attributes
        self.num_vertices = self.surface.vertices.shape[0]
        self.update_surface_attributes()

        print("Brain extractor initialization complete!")

    def rebuild_surface(self, vertices: np.ndarray):
        """
            Rebuilds the surface mesh for given updated vertices
        """
        self.surface = trimesh.Trimesh(vertices=vertices, faces=self.surface.faces)
        self.update_surface_attributes()

    @staticmethod
    @jit(nopython=True, cache=True)
    def update_surf_attr(vertices, normals, neighbors_idx):
        # get normals as contiguous in memory
        normals = np.ascontiguousarray(normals)
        
        # the neighbors array is tricky because it doesn't 
        # have the structure of a nice rectangular array
        # we initialize it to be the largest size (6) then we
        # can make a check for valid vertices later with neighbors size
        neighbors = np.zeros((vertices.shape[0], 6, 3))
        neighbors_size = np.zeros(vertices.shape[0], dtype=np.int8)
        for i, ni in enumerate(neighbors_idx):
            for j, vi in enumerate(ni):
                neighbors[i,j,:] = vertices[vi]
            neighbors_size[i] = j + 1
        
        # compute centroids
        centroids = np.zeros((vertices.shape[0], 3))
        for i, (n, s) in enumerate(zip(neighbors, neighbors_size)):
            centroids[i,0] = np.mean(n[:s,0])
            centroids[i,1] = np.mean(n[:s,1])
            centroids[i,2] = np.mean(n[:s,2])

        # return optimized surface attributes
        return normals, neighbors, neighbors_size, centroids

    def update_surface_attributes(self):
        """
            Updates attributes related to the surface
        """
        self.vertices = np.array(self.surface.vertices)
        self.vertex_neighbors_idx = List([np.array(i) for i in self.surface.vertex_neighbors])
        # self.vertex_neighbors_idx = List(self.surface.vertex_neighbors) 
        self.vertex_normals, self.vertex_neighbors, self.vertex_neighbors_size, \
            self.vertex_neighbors_centroids = self.update_surf_attr(
                self.vertices,
                self.surface.vertex_normals,
                self.vertex_neighbors_idx)
        # self.vertex_neighbors = [np.vstack([self.vertices[v] for v in ni]) for ni in self.vertex_neighbors_idx]
        # self.vertex_neighbors_centroids = np.vstack([np.mean(self.vertex_neighbors[i], axis=0) for i in range(self.num_vertices)])
        # self.vertex_normals = np.ascontiguousarray(self.surface.vertex_normals)
        # breakpoint()

    def run(self, iterations=1000):
        """
            Runs the extraction step.

            This deforms the surface based on the method outlined in"

            Smith SM. Fast robust automated brain extraction. Hum Brain Mapp.
            2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568;
            PMCID: PMC6871816.

        """
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

        # surface deformation loop
        for i in range(iterations):
            print("Iteration: %d" % i)
            # update the mean intervertex distances at intervals of 100 (and iteration 50)
            if i % 100 == 0 or i == 50:
                # l = self.get_mean_intervertex_distance()
                l = self.get_mean_intervertex_distance(
                    self.vertices,
                    self.vertex_neighbors,
                    self.vertex_neighbors_size
                )
                # print(l)
                # breakpoint()
            # run one step of deformation
            self.step_of_deformation(
                self.data, self.vertices, self.vertex_normals,
                self.vertex_neighbors_centroids,
                l, self.t2, self.t, self.tm, self.t98,
                self.E, self.F, self.bt, self.d1, self.d2,
                s_vectors, s_n, s_t, u1, u2, u3, u
            )
            # update the surface
            self.rebuild_surface(self.vertices + u)

    @staticmethod
    @jit(nopython=True)
    def step_of_deformation(
        data: np.ndarray,
        vertices: np.ndarray,
        normals: np.ndarray,
        neighbors_centroids: np.ndarray,
        l: float,
        t2: float,
        t: float,
        tm: float,
        t98:float,
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
        u: np.ndarray
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
            u1[i] = 0.5*s_t[i]

            # compute local radius of curvature
            r = (l**2)/(2*l2norm(s_n[i]))

            # compute f2
            f2 = (1 + np.tanh(F*(1/r - E)))/2

            # set component u2
            u2[i] = f2*s_n[i]

            # get endpoints directed interior (distance set by d1 and d2)
            e1 = closest_integer_point(vertex - d1 * normals[i])
            e2 = closest_integer_point(vertex - d2 * normals[i])

            # get lines created by e1/e2
            c = closest_integer_point(vertex)
            i1 = bresenham3d(c, e1)
            i2 = bresenham3d(c, e2)

            # get Imin/Imax
            linedata1 = [data[d[0],d[1],d[2]] for d in i1]
            linedata1.append(tm)
            linedata1 = np.array(linedata1)
            Imin = np.max(np.array([t2, np.min(linedata1)]))
            linedata2 = [data[d[0],d[1],d[2]] for d in i2]
            linedata2.append(t)
            linedata2 = np.array(linedata2)
            Imax = np.min(np.array([tm, np.max(linedata2)]))

            # get tl
            tl = (Imax - t2)*bt + t2

            # compute f3
            f3 = 0.05*2*(Imin - tl)/(Imax - t2)*l

            # get component u3
            u3[i] = f3 * normals[i]

        # get displacement vector
        u[:,:] = u1 + u2 + u3

    # def get_mean_intervertex_distance(self):
    #     """
    #         Computes the mean intervertex distance across the entire surface
    #     """
    #     # Compute the mean intervertex distance
    #     return np.mean([self.compute_mlid(self.vertices[i] - self.vertex_neighbors[i]) for i in range(self.num_vertices)])

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_mean_intervertex_distance(vertices: np.ndarray, neighbors: np.ndarray, sizes: np.ndarray):
        """
            Computes the mean intervertex distance across the entire surface
        """
        mivd = np.zeros(vertices.shape[0])
        for v in range(vertices.shape[0]):
            vecs = vertices[v] - neighbors[v,:sizes[v]]
            vd = np.zeros(vecs.shape[0])
            for i in range(vecs.shape[0]):
                vd[i] = l2norm(vecs[i])
            mivd[v] = np.mean(vd)
        return np.mean(mivd)

    # @staticmethod
    # @jit(nopython=True, cache=True)
    # def compute_mlid(vecs: np.ndarray):
    #     """
    #         Computes the mean local intervertex distance
    #     """
    #     result = list()
    #     for i in range(vecs.shape[0]): # pylint: disable=not-an-iterable
    #         result.append(l2norm(vecs[i]))
    #     return np.mean(np.array(result))

    def compute_mask(self):
        """
            Convert surface mesh to volume
        """
        if not hasattr(self, "mask"):
            self.mask = find_enclosure(self.surface, self.shape)
        return self.mask

    def save_mask(self, filename):
        """
            Saves brain extraction to nifti file
        """
        mask = self.compute_mask()
        nib.Nifti1Image(mask, self.img.affine).to_filename(filename)

    def save_surface(self, filename):
        """
            Save surface in .stl 
        """
        self.surface.export(filename)
