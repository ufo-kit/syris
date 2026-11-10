import abc
import logging
import operator
import pathlib
import typing

import numpy as np
import pyvista as pv
import quantities as pq
import vtk
from vtk.util.numpy_support import vtk_to_numpy

LOG = logging.getLogger(__name__)


def renderMesh(polydata):
    plotter = pv.Plotter()
    plotter.add_mesh(polydata)
    plotter.show()


def extractContiguousTriangles(polydata):
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

        faces = np.hstack(
            (
                np.full((num_triangles, 1), 3),
                np.arange(num_points).reshape(num_triangles, 3),
            )
        ).flatten()

        polydata = pv.PolyData(verts_magnitude, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(polydata)
        plotter.show()


class PyvistaReader(MeshReaderBase):
    def __init__(
        self,
        filename: typing.Union[str, pathlib.Path, pv.DataSet],
        unit: pq.Quantity = pq.m,
        dtype=np.float32,
        compute_normals=False,
        repair=False,
        float32_rounding=False,
        **kwargs,
    ):
        """
        Read a mesh file using PyVista and convert it to a format suitable for rendering.

        Parameters
        ----------
        filename : str
            The path to the mesh file.
        unit : pq.Quantity, optional"
        repair : bool, optional
            Resolve self-intersections of a closed, oriented material boundary
            with the optional libigl dependency. Preserve cavity orientation.
        float32_rounding : bool, optional
            Experimental bounded rounding of repaired coordinates to Float32.
            Requires repair=True and dtype=np.float32; see repair_mesh.
        """
        super().__init__()
        if float32_rounding and (not repair or np.dtype(dtype) != np.dtype(np.float32)):
            raise ValueError("float32_rounding requires repair=True and dtype=np.float32.")

        if isinstance(filename, (str, pathlib.Path)):
            self.filename = str(filename)
            mesh = pv.read(self.filename)
        elif isinstance(filename, pv.DataSet):
            self.filename = "in-memory-mesh"
            mesh = filename
        else:
            raise TypeError(
                f"Expected a filename (str or Path) or a PyVista object, but got {type(filename)}."
            )

        # Handle MultiBlock datasets
        while isinstance(mesh, pv.MultiBlock):
            mesh = mesh[0]

        if not mesh.is_all_triangles:
            LOG.debug("Mesh is not all triangles. Triangulating...")
            mesh = mesh.triangulate(inplace=False, progress_bar=True)
        else:
            LOG.debug("Mesh is already all triangles.")

        if repair:
            from .meshrepair import repair_mesh

            points, triangles = repair_mesh(
                mesh.points,
                np.asarray(mesh.faces).reshape(-1, 4)[:, 1:],
                dtype=dtype,
                float32_rounding=float32_rounding,
            )
            faces = np.column_stack((np.full(len(triangles), 3), triangles))
            mesh = pv.PolyData(points, faces)
            # Recompute from repaired face winding. Auto-orientation of separate
            # components would reverse inward-facing cavity boundaries.
            verts = points[triangles].astype(np.float64)
            normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            mesh.cell_data["Normals"] = normals.astype(dtype)
            mesh.cell_data.active_normals_name = "Normals"

        self.polydata = mesh

        if compute_normals and mesh.cell_normals is None:
            LOG.debug("Computing cell normals...")
            mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                inplace=False,
                progress_bar=True,
            )

        LOG.debug("Copying mesh data to contiguous NumPy arrays...")
        points = np.array(mesh.points, copy=True, dtype=dtype)
        faces_vtk = np.array(mesh.faces, copy=True)

        # Handle potentially empty normals
        cell_normals = (
            np.array(mesh.cell_normals, copy=True, dtype=dtype)
            if mesh.cell_normals is not None
            else np.array([])
        )
        bounds = np.array(mesh.bounds, copy=True, dtype=dtype)

        triangles = faces_vtk.reshape(-1, 4)[:, 1:]

        triangle_vertices = points[triangles]
        triangle_vertices = triangle_vertices.flatten().reshape(-1, 3)

        if triangles.size > 0:
            triangle_verts = points[triangles]
            edge0_sq_len = np.sum(
                (triangle_verts[:, 1, :] - triangle_verts[:, 0, :]) ** 2,
                axis=1,
            )
            edge1_sq_len = np.sum(
                (triangle_verts[:, 2, :] - triangle_verts[:, 1, :]) ** 2,
                axis=1,
            )
            edge2_sq_len = np.sum(
                (triangle_verts[:, 0, :] - triangle_verts[:, 2, :]) ** 2,
                axis=1,
            )
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
        self.dynamic_range = 1
        if self._smallest_feature_size > 0 and np.isfinite(self._smallest_feature_size):
            self.dynamic_range = self._largest_feature_size / self._smallest_feature_size
            LOG.debug(f"Dynamic Range: {self.dynamic_range:.2f} : 1")
        else:
            LOG.debug("Dynamic Range: N/A (smallest feature is zero or invalid)")

        # Store the final NumPy arrays
        self._vertices = triangle_vertices.T.copy().astype(dtype) * unit
        self._triangles = triangles
        self._normals = cell_normals * unit
        self._bounds = bounds * unit

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


class MeshAnimationReader(MeshReaderBase):
    """Read an OBJ folder or a temporal VTKHDF polygon mesh one frame at a time.

    ``source`` is either a directory of lexically ordered ``.obj`` files or a
    ``.vtkhdf``/``.hdf`` file containing ``vtkPolyData``. A static VTKHDF mesh
    is a one-frame sequence. VTKHDF timestamps are available as ``time_values``;
    untimed sources have ``time_values=None``. No interpolation is performed.

    Each iteration yields this reader with updated ``vertices`` (3*N, 3),
    ``normals`` (N, 3), ``bounds`` (6,), ``frame_index`` and ``frame_time``.
    Vertices and bounds use the file's coordinate units, without a quantities
    wrapper; normals are dimensionless.
    Float32 remains the default; pass ``dtype=np.float64`` to retain Float64
    coordinates. Copy the arrays to retain snapshots of the mutable reader.
    """

    def __init__(self, source: typing.Union[str, pathlib.Path], *, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Animation dtype must be float32 or float64.")

        self._vertices = None
        self._normals = None
        self._bounds = None
        self.frame_index = None
        self.frame_time = None
        self.time_values = None
        self.filenames = []
        self._hdf_reader = None

        source = pathlib.Path(source)
        if source.is_dir():
            self.filenames = sorted(str(p) for p in source.glob("*.obj") if p.is_file())
            if not self.filenames:
                raise FileNotFoundError(f"No .obj files found in folder: {source}")
        else:
            if not source.is_file():
                raise FileNotFoundError(source)
            if source.suffix.lower() not in (".vtkhdf", ".hdf"):
                raise ValueError("Expected an OBJ folder or a .vtkhdf/.hdf file.")
            if not hasattr(vtk, "vtkHDFReader"):
                raise ImportError("VTKHDF animations require a VTK build with vtkHDFReader.")
            self.filename = str(source)
            self._hdf_reader = vtk.vtkHDFReader()
            self._hdf_reader.SetFileName(self.filename)
            if not self._hdf_reader.CanReadFile(self.filename):
                raise ValueError(f"Not a readable VTKHDF file: {source}")
            self._update(self._hdf_reader, self.filename, information_only=True)
            if not isinstance(self._hdf_reader.GetOutputDataObject(0), vtk.vtkPolyData):
                raise ValueError("VTKHDF animations must contain a vtkPolyData surface mesh.")
            self._number_of_frames = max(1, self._hdf_reader.GetNumberOfSteps())
            times = self._hdf_reader.GetOutputInformation(0).Get(
                vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
            )
            if times is not None:
                self.time_values = tuple(times)
                if len(self.time_values) != self._number_of_frames:
                    raise ValueError("VTKHDF frame count does not match its timestamps.")

    @staticmethod
    def _update(reader, filename, *, information_only=False):
        # VTK normally reports I/O errors as events, without raising in Python.
        errors = []
        observer = reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
        try:
            if information_only:
                reader.UpdateInformation()
            else:
                reader.Update()
        finally:
            reader.RemoveObserver(observer)
        if errors or reader.GetErrorCode():
            raise OSError(f"Failed to read mesh data from {filename}")

    def __len__(self):
        if self._hdf_reader is None:
            # Existing benchmark callers may reorder or subset this public list.
            return len(self.filenames)
        return self._number_of_frames

    def __iter__(self):
        return self.iter_frames()

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
        reader.SetFileName(str(filename))
        self._update(reader, filename)
        return self._extract_frame(reader.GetOutput())

    def _extract_frame(self, polydata):
        if not isinstance(polydata, vtk.vtkPolyData):
            raise ValueError("Animation frames must contain a vtkPolyData surface mesh.")
        if polydata.GetNumberOfVerts() or polydata.GetNumberOfLines():
            raise ValueError(
                "Animation frames must contain only surface cells, not points or lines."
            )
        if not (polydata.GetNumberOfPolys() or polydata.GetNumberOfStrips()):
            raise ValueError("Animation frame contains no surface cells.")

        if polydata.GetNumberOfStrips() or polydata.GetPolys().IsHomogeneous() != 3:
            triangulate = vtk.vtkTriangleFilter()
            triangulate.SetInputData(polydata)
            triangulate.Update()
            polydata = triangulate.GetOutput()
        if not polydata.GetNumberOfPolys():
            raise ValueError("Animation frame contains no triangles after triangulation.")

        # Extract and process data
        points_np = vtk_to_numpy(polydata.GetPoints().GetData())
        cells_np = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())

        vertices = points_np[cells_np].astype(self.dtype, copy=False)
        bounds = np.array(polydata.GetBounds(), dtype=self.dtype)

        # VTKHDF can preserve the array without preserving its active-normal role.
        cell_normals = polydata.GetCellData().GetNormals()
        if cell_normals is None:
            cell_normals = polydata.GetCellData().GetArray("Normals")
        if cell_normals is None:
            normal_filter = vtk.vtkPolyDataNormals()
            normal_filter.SetInputData(polydata)
            normal_filter.ComputeCellNormalsOn()
            normal_filter.ComputePointNormalsOff()
            normal_filter.Update()
            cell_normals = normal_filter.GetOutput().GetCellData().GetNormals()
        if (
            cell_normals.GetNumberOfComponents() != 3
            or cell_normals.GetNumberOfTuples() != polydata.GetNumberOfPolys()
        ):
            raise ValueError("Animation cell normals must have shape (number of triangles, 3).")
        # Own the arrays: updating the persistent HDF reader may reuse VTK buffers.
        normals = np.array(vtk_to_numpy(cell_normals), dtype=self.dtype, copy=True)

        return vertices, normals, bounds

    def read_frame(self, index):
        """Load a zero-based frame index and return this updated reader."""
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(f"Frame index {index} is outside [0, {len(self)}).")
        if self._hdf_reader is None:
            vertices, normals, bounds = self._read_file(self.filenames[index])
        else:
            self._hdf_reader.SetStep(index)
            self._update(self._hdf_reader, self.filename)
            vertices, normals, bounds = self._extract_frame(self._hdf_reader.GetOutputDataObject(0))

        self._vertices = vertices
        self._normals = normals
        self._bounds = bounds
        self.frame_index = index
        self.frame_time = self.time_values[index] if self.time_values is not None else None
        return self

    def iter_frames(self):
        """Yield this reader for each frame; repeated iteration restarts at zero."""
        for index in range(len(self)):
            yield self.read_frame(index)


class WavefrontAnimationReader(MeshAnimationReader):
    """Backward-compatible name for :class:`MeshAnimationReader`.

    ``folder`` retains its original keyword name and also accepts VTKHDF files.
    """

    def __init__(self, folder: typing.Union[str, pathlib.Path], *, dtype=np.float32):
        super().__init__(folder, dtype=dtype)


class RandomMeshReader(MeshReaderBase):
    def __init__(self, n: int, eps: float, lengths: np.ndarray, origin: np.ndarray):
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
        self._bounds = np.array(
            [
                origin[0] - lengths[0] / 2,
                origin[0] + lengths[0] / 2,
                origin[1] - lengths[1] / 2,
                origin[1] + lengths[1] / 2,
                origin[2] - lengths[2] / 2,
                origin[2] + lengths[2] / 2,
            ]
        ).astype(np.float32)

    @property
    def scene(self):
        return [self.vertices, self.normals, self.bounds]

    def visualize(self, normals=False, figsize=(800, 1024)):
        plotter = pv.Plotter(window_size=figsize)
        plotter.add_mesh(self.polydata)
        plotter.add_points(self.points, color="red")
        if normals:
            plotter.add_arrows(self.points, self.normals, mag=0.1)
        plotter.show()
