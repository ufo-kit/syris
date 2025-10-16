import numpy as np
import pyvista as pv
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import quantities as pq
import abc
import logging

LOG = logging.getLogger(__name__)

def renderMesh (polydata):
    plotter = pv.Plotter()
    plotter.add_mesh(polydata)
    plotter.show()

def extractContiguousTriangles (polydata):
    points = polydata.GetPoints()
    cells = polydata.GetPolys()

    # Convert to numpy
    points_np = vtk_to_numpy(points.GetData())
    cells_np = vtk_to_numpy(cells.GetData())

    # The first element is the number of vertices
    # Verify the number of vertices
    cells_np = cells_np.reshape(-1, 4)
    if np.any(cells_np[:, 0] != 3):
        raise Exception("Only triangles are supported")
    cells_np = cells_np[:, 1:].flatten()
    vertices = points_np[cells_np]

    if len(vertices) != len(cells_np):
        raise Exception("Vertices and cells do not match")
    
    return vertices, cells_np

class MeshReaderBase(abc.ABC):
    """
    Abstract interface for all mesh readers.
    Ensures that any subclass will have the required properties.
    """
    @property
    @abc.abstractmethod
    def vertices(self):
        """Returns the mesh vertices as a NumPy array with units."""
        pass

    @property
    @abc.abstractmethod
    def normals(self):
        """Returns the cell normals as a NumPy array."""
        pass

    @property
    @abc.abstractmethod
    def bounds(self):
        """Returns the mesh bounds as a NumPy array with units."""
        pass

    def visualize(self):
        """
        Provides a default visualization method by reconstructing a
        PolyData object from the vertices. Subclasses can override this
        for better performance if they already have a PolyData object.
        """
        verts = self.vertices
        
        verts_magnitude = verts.rescale(pq.m).magnitude

        num_points = len(verts_magnitude)
        if num_points % 3 != 0:
            raise ValueError("Vertex data does not represent a valid triangle mesh.")
        
        num_triangles = num_points // 3
        
        faces = np.hstack((
            np.full((num_triangles, 1), 3),
            np.arange(num_points).reshape(num_triangles, 3)
        )).flatten()

        polydata = pv.PolyData(verts_magnitude, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(polydata)
        plotter.show()


class PyvistaReader(MeshReaderBase):
    def __init__(self, filename : str, unit : pq.Quantity = pq.m, dtype=np.float32, compute_normals=False, triangulate=True, **kwargs):
        """
        Read a mesh file using PyVista and convert it to a format suitable for rendering.

        Parameters
        ----------
        filename : str
            The path to the mesh file.
        unit : pq.Quantity, optional"
        """
        super().__init__()
        self.filename = filename

        mesh = pv.read(self.filename)

        # Handle MultiBlock datasets
        while isinstance(mesh, pv.MultiBlock):
            mesh = mesh[0]
        
        # Ensure the mesh is triangulated
        if 3 * mesh.n_cells != mesh.n_points:
            mesh = mesh.triangulate(inplace=False, progress_bar=True)

        self.polydata = mesh

        # Ensure the normals are calculated
        if mesh.cell_normals is None and compute_normals:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False, progress_bar=True)

        triangles = mesh.faces.reshape(-1, 4)[:, 1:]

        points = mesh.points
        triangle_vertices = points[triangles]
        triangle_vertices = triangle_vertices.flatten().reshape(-1, 3)

        if triangles.size > 0:
            # Step 1: Get all triangle vertex coordinates
            triangle_verts = points[triangles]
            
            # Step 2: Calculate all squared edge lengths
            edge0_sq_len = np.sum((triangle_verts[:, 1, :] - triangle_verts[:, 0, :])**2, axis=1)
            edge1_sq_len = np.sum((triangle_verts[:, 2, :] - triangle_verts[:, 1, :])**2, axis=1)
            edge2_sq_len = np.sum((triangle_verts[:, 0, :] - triangle_verts[:, 2, :])**2, axis=1)

            # Step 3: Find the minimum of all NON-ZERO squared lengths
            all_sq_lens = np.concatenate([edge0_sq_len, edge1_sq_len, edge2_sq_len])
            non_zero_sq_lens = all_sq_lens[all_sq_lens > 0]

            if non_zero_sq_lens.size > 0:
                min_edge_length_sq = np.min(non_zero_sq_lens)
                max_edge_length_sq = np.max(non_zero_sq_lens)
                smallest_feature = np.sqrt(min_edge_length_sq)
                largest_feature = np.sqrt(max_edge_length_sq)
            else:
                LOG.debug("WARNING: Mesh appears to have no edges with a length > 0.")
                smallest_feature = np.inf
                largest_feature = np.inf
        else:
            smallest_feature = np.inf
            largest_feature = np.inf

        self._smallest_feature_size = smallest_feature.astype(dtype)
        self._largest_feature_size = largest_feature.astype(dtype)        
        
        if self._smallest_feature_size > 0 and np.isfinite(self._smallest_feature_size):
            dynamic_range = self._largest_feature_size / self._smallest_feature_size
            LOG.debug(f"Dynamic Range: {dynamic_range:.2f} : 1")
        else:
            LOG.debug("Dynamic Range: N/A (smallest feature is zero or invalid)")

        self._vertices = np.array(triangle_vertices).T.astype(dtype) * unit
        self._triangles = triangles
        self._normals = np.array(mesh.cell_normals).astype(dtype) * unit
        self._bounds = np.array(mesh.bounds).astype(dtype) * unit

    @property
    def vertices(self):
        return self._vertices

    @property
    def normals(self):
        return self._normals

    @property
    def bounds(self):
        return self._bounds
    
    @property
    def epsilon(self):
        return self._smallest_feature_size


class WavefrontAnimationReader(MeshReaderBase):
    def __init__(self, folder: str):
        self.filenames = sorted([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.obj')
        ])
        if not self.filenames:
            raise FileNotFoundError(f"No .obj files found in folder: {folder}")
        
        # Initialize properties to None; they will be set by the iterator
        self._vertices = None
        self._normals = None
        self._bounds = None

    @property
    def vertices(self):
        if self._vertices is None:
            raise AttributeError("Vertices not loaded. Iterate over the reader first.")
        return self._vertices

    @property
    def normals(self):
        if self._normals is None:
            raise AttributeError("Normals not loaded. Iterate over the reader first.")
        return self._normals

    @property
    def bounds(self):
        if self._bounds is None:
            raise AttributeError("Bounds not loaded. Iterate over the reader first.")
        return self._bounds

    def _read_file(self, filename):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        
        # Extract and process data
        points_np = vtk_to_numpy(polydata.GetPoints().GetData())
        cells_np = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:].flatten()
        
        vertices = points_np[cells_np].astype(np.float32)
        bounds = np.array(polydata.GetBounds()).astype(np.float32)
        
        # Compute normals if not present
        if polydata.GetCellData().GetNormals() is None:
            normal_filter = vtk.vtkPolyDataNormals()
            normal_filter.SetInputData(polydata)
            normal_filter.ComputeCellNormalsOn()
            normal_filter.ComputePointNormalsOff()
            normal_filter.Update()
            normals = vtk_to_numpy(normal_filter.GetOutput().GetCellData().GetNormals()).astype(np.float32)
        else:
            normals = vtk_to_numpy(polydata.GetCellData().GetNormals()).astype(np.float32)
            
        return vertices, normals, bounds

    def iter_frames(self):
        """Generator that yields the data for each frame in the animation sequence."""
        for filename in self.filenames:
            # Read the data for the current frame
            v, n, b = self._read_file(filename)
            
            # Update instance properties so they can be accessed after iteration
            self._vertices = v
            self._normals = n
            self._bounds = b
            
            yield self # Yield the reader instance itself


class RandomMeshReader(MeshReaderBase):
    def __init__(self, n : int, eps : float, lengths : np.ndarray, origin : np.ndarray):
        super().__init__()

        rng = np.random.default_rng()
        
        self.points = rng.random((n, 3)) * lengths + origin - lengths / 2
        
        # Create an empty array for vertices
        self._vertices = np.empty((n * 3, 3)).astype(np.float32)

        # Interleave u, v, and w without using a for loop
        self._vertices[0::3] = self.points
        self._vertices[1::3] = self.points + rng.random((n, 3)) * eps
        self._vertices[2::3] = self.points + rng.random((n, 3)) * eps

        self.triangles = np.arange((n * 3)).reshape(-1, 3)
        poly_faces = np.hstack((np.full((n, 1), 3), self.triangles)).flatten()
        self.polydata = pv.PolyData(self.vertices, poly_faces)

        # New random seed
        rng = np.random.default_rng()
        self._normals = np.random.rand(n, 3).astype(np.float32) - 0.5
        self._bounds = np.array([origin[0] - lengths[0] / 2, origin[0] + lengths[0] / 2, origin[1] - lengths[1] / 2, origin[1] + lengths[1] / 2, origin[2] - lengths[2] / 2, origin[2] + lengths[2] / 2]).astype(np.float32)
    
    @property
    def scene (self):
        return [self.vertices, self.normals, self.bounds]
    
    def visualize (self, normals=False, figsize=(800, 1024)):
        plotter = pv.Plotter(window_size=figsize)
        plotter.add_mesh(self.polydata)
        plotter.add_points(self.points, color='red')
        if normals:
            plotter.add_arrows(self.points, self.normals, mag=0.1)
        plotter.show()
    
    