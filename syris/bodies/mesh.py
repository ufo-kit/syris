# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""Bodies made from mesh."""
import itertools
import re
import numpy as np
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import quantities as q
import syris.config as cfg
import syris.geometry as geom
import syris.gpu.util as gutil
from syris.bodies.base import MovableBody
from syris.devices.cameras import Camera
from syris.util import get_magnitude, make_tuple
import cupy as cp
import pyvista as pv
import os, vtk
from vtk.util.numpy_support import vtk_to_numpy
from itertools import islice


class Mesh(MovableBody):

    """Rigid Body based on *triangles* which form a polygon mesh. The triangles are a 2D array with
    shape (3, N), where N / 3 is the number of triangles. One polygon is formed by three consecutive
    triangles, e.g. when::

        triangles =[[A0, A1, A2],
                    [B0, B1, B2],
                    [C0, C1, C2],
                    ...]

    then A, B, C are one triangle's points. *center* determines the center of the local
    coordinates, it can be one of None, 'bbox', 'gravity' or a (x, y, z) tuple specifying an
    arbitrary point.
    """

    def __init__(
        self,
        triangles,
        trajectory,
        material=None,
        orientation=geom.Y_AX,
        unit=q.m,
        center="bbox",
        normals=None,
        bounds=None,
        iterations=1,
        coordinate_system=None,
    ):
        """Constructor."""
        if center is None:
            point = (0, 0, 0) * unit
        else:
            # Arbitrary point
            point = center

        self._unit = unit
        point = np.insert(point.rescale(self._unit).magnitude, 3, 0)[:, np.newaxis]

        self._triangles = triangles
        self._furthest_point = np.max(np.sqrt(np.sum(self._triangles ** 2, axis=0)))
        self._iterations = iterations

        if normals is None:
            # TODO: make projection function that works without normals
            pass
        else:
            self._normals = normals

        if bounds is None:
            # TODO: Compute bounds (xmin, xmax, ymin, ymax, zmin, zmax)
            pass
        else:
            self._bounds = bounds

        self._current = np.copy(self._triangles)

        self._tree = None

        # Build the tree
        self._build_tree()

        super(Mesh, self).__init__(trajectory, material=material, orientation=orientation, coordinate_system=coordinate_system)

    @property
    def furthest_point(self):
        """Furthest point from the center."""
        return self._furthest_point * self._unit

    @property
    def bounding_box(self):
        """Bounding box implementation."""
        x, y, z = self.extrema

        return geom.BoundingBox(geom.make_points(x, y, z))

    @property
    def num_triangles(self):
        """Number of triangles in the mesh."""
        return self._current.shape[1] // 3

    @property
    def extrema(self):
        """Mesh extrema as ((x_min, x_max), (y_min, y_max), (z_min, z_max))."""
        return (
            (self._compute(min, 0), self._compute(max, 0)),
            (self._compute(min, 1), self._compute(max, 1)),
            (self._compute(min, 2), self._compute(max, 2)),
        ) * self._unit

    @property
    def center_of_gravity(self):
        """Get body's center of gravity as (x, y, z)."""
        center = (self._compute(np.mean, 0), self._compute(np.mean, 1), self._compute(np.mean, 2))

        return np.array(center) * self._unit

    @property
    def center_of_bbox(self):
        """The bounding box center."""

        def get_middle(ends):
            return (ends[0] + ends[1]) / 2.0

        return np.array([get_middle(ends) for ends in self.extrema.magnitude]) * self._unit

    @property
    def diff(self):
        """Smallest and greatest difference between all mesh points in all three dimensions. Returns
        ((min(dx), max(dx)), (min(dy), max(dy)), (min(dz), max(dz))).
        """
        def min_nonzero(ar):
            return min(ar[np.where(ar != 0)])

        def max_nonzero(ar):
            return max(ar[np.where(ar != 0)])

        def func(ar):
            return np.abs(ar[1:] - ar[:-1])

        x_diff = self._compute(func, 0)
        y_diff = self._compute(func, 1)
        z_diff = self._compute(func, 2)

        return (
            (min_nonzero(x_diff), max_nonzero(x_diff)),
            (min_nonzero(y_diff), max_nonzero(y_diff)),
            (min_nonzero(z_diff), max_nonzero(z_diff)),
        ) * self._unit

    @property
    def vectors(self):
        """The triangles as B - A and C - A vectors where A, B, C are the triangle vertices. The
        result is transposed, i.e. axis 1 are x, y, z coordinates.
        """
        a = self._current[:-1, 0::3]
        b = self._current[:-1, 1::3]
        c = self._current[:-1, 2::3]
        v_0 = (b - a).transpose()
        v_1 = (c - a).transpose()

        return v_0 * self._unit, v_1 * self._unit

    @property
    def areas(self):
        """Triangle areas."""
        v_0, v_1 = self.vectors
        cross = np.cross(v_0, v_1)

        return np.sqrt(np.sum(cross * cross, axis=1)) / 2 * self._unit ** 2

    @property
    def normals(self):
        """Triangle normals."""
        v_0, v_1 = self.vectors

        return np.cross(v_0, v_1) * self._unit

    @property
    def max_triangle_x_diff(self):
        """Get the greatest x-distance between triangle vertices."""
        x_0 = self._current[0, 0::3]
        x_1 = self._current[0, 1::3]
        x_2 = self._current[0, 2::3]
        d_0 = np.max(np.abs(x_1 - x_0))
        d_1 = np.max(np.abs(x_1 - x_2))
        d_2 = np.max(np.abs(x_2 - x_1))

        return max(d_0, d_1, d_2) * self._unit

    @property
    def triangles(self):
        """Return current triangle mesh."""
        return self._current[:-1, :] * self._unit

    def sort(self):
        """Sort triangles based on the greatest x-coordinate in an ascending order. Also sort
        vertices inside the triangles so that the greatest one is the last one, however, the
        position of the two remaining ones is not sorted.
        """
        # Extract x-coordinates
        x = self._current[0, :].reshape(self.num_triangles, 3)
        # Get vertices with the greatest x-coordinate and scale the indices up so we can work with
        # the original array
        factor = np.arange(self.num_triangles) * 3
        representatives = np.argmax(x, axis=1) + factor
        # Get indices which sort the triangles
        base = 3 * np.argsort(self._current[0, representatives])
        indices = np.empty(3 * len(base), dtype=int)
        indices[::3] = base
        indices[1::3] = base + 1
        indices[2::3] = base + 2

        # Sort the triangles such that the largest x-coordinate is in the last vertex
        tmp = np.copy(self._current[:, 2::3])
        self._current[:, 2::3] = self._current[:, representatives]
        self._current[:, representatives] = tmp

        # Sort the triangles among each other
        self._current = self._current[:, indices]

    def get_degenerate_triangles(self, eps=1e-3 * q.deg):
        """Get triangles which are close to be parallel with the ray in z-direction based on the
        current transformation matrix. *eps* is the tolerance for the angle between a triangle and
        the ray to be still considered parallel.
        """
        ray = np.array([0, 0, 1]) * self._unit
        dot = np.sqrt(np.sum(self.normals ** 2, axis=1))
        theta = np.arccos(np.dot(self.normals, ray) / dot)
        diff = np.abs(theta - np.pi / 2 * q.rad)
        indices = np.where(diff < eps)[0]

        # Stretch to vertex indices
        t_indices = np.empty(3 * len(indices), dtype=int)
        for i in range(3):
            t_indices[i::3] = 3 * indices + i
        close = self._current[:-1, t_indices]

        return close * self._unit

    def _compute(self, func, axis):
        """General function for computations with triangles."""
        return func(self._current[axis, :])

    def _make_vertices(self, index, pixel_size):
        """Make a flat array of vertices belong to *triangles* at *index*."""
        # Convert to meters
        vertices = self._current[:, index::3] / pixel_size.rescale(self._unit).magnitude

        return vertices.transpose().flatten().astype(cfg.PRECISION.np_float)

    def _make_inputs(self, queue, pixel_size):
        v_1 = cl_array.to_device(queue, self._make_vertices(0, pixel_size[1]))
        v_2 = cl_array.to_device(queue, self._make_vertices(1, pixel_size[0]))
        v_3 = cl_array.to_device(queue, self._make_vertices(2, pixel_size[1]))

        return v_1, v_2, v_3

    def transform(self):
        """Apply transformation *matrix* and return the resulting triangles."""
        matrix = self.get_rescaled_transform_matrix(self._unit)
        self._current = np.dot(matrix.astype(self._triangles.dtype), self._triangles)

    # def _project(self, shape, pixel_size, offset, t=None, queue=None, out=None, block=False):
    #     """Projection implementation."""

    #     def get_crop(index, fov):
    #         minimum = max(self.extrema[index][0], fov[index][0])
    #         maximum = min(self.extrema[index][1], fov[index][1])

    #         return minimum - offset[::-1][index], maximum - offset[::-1][index]

    #     def get_px_value(value, round_func, ps):
    #         return int(round_func(get_magnitude(value / ps)))

    #     # Move to the desired location, apply the T matrix and resort the triangles
    #     self.transform()
    #     self.sort()

    #     psm = pixel_size.simplified.magnitude
    #     fov = offset + shape * pixel_size
    #     fov = (
    #         np.concatenate((offset.simplified.magnitude[::-1], fov.simplified.magnitude[::-1]))
    #         .reshape(2, 2)
    #         .transpose()
    #         * q.m
    #     )
    #     if out is None:
    #         out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_float)

    #     if (
    #         self.extrema[0][0] < fov[0][1]
    #         and self.extrema[0][1] > fov[0][0]
    #         and self.extrema[1][0] < fov[1][1]
    #         and self.extrema[1][1] > fov[1][0]
    #     ):
    #         # Object inside FOV
    #         x_min, x_max = get_crop(0, fov)
    #         y_min, y_max = get_crop(1, fov)
    #         x_min_px = get_px_value(x_min, np.floor, pixel_size[1])
    #         x_max_px = get_px_value(x_max, np.ceil, pixel_size[1])
    #         y_min_px = get_px_value(y_min, np.floor, pixel_size[0])
    #         y_max_px = get_px_value(y_max, np.ceil, pixel_size[0])
    #         width = min(x_max_px - x_min_px, shape[1])
    #         height = min(y_max_px - y_min_px, shape[0])
    #         compute_offset = cltypes.make_int2(x_min_px, y_min_px)
    #         v_1, v_2, v_3 = self._make_inputs(queue, pixel_size)
    #         max_dx = self.max_triangle_x_diff.simplified.magnitude / psm[1]
    #         # Use the same pixel size as for the x-axis, which will work for objects "not too far"
    #         # from the imaging plane
    #         min_z = self.extrema[2][0].simplified.magnitude / psm[1]
    #         offset = gutil.make_vfloat2(*(offset / pixel_size).simplified.magnitude[::-1])

    #         ev = cfg.OPENCL.programs["mesh"].compute_thickness(
    #             queue,
    #             (width, height),
    #             None,
    #             v_1.data,
    #             v_2.data,
    #             v_3.data,
    #             out.data,
    #             np.int32(self.num_triangles),
    #             np.int32(shape[1]),
    #             compute_offset,
    #             offset,
    #             cfg.PRECISION.np_float(psm[1]),
    #             cfg.PRECISION.np_float(max_dx),
    #             cfg.PRECISION.np_float(min_z),
    #             np.int32(self.iterations),
    #         )
    #         if block:
    #             ev.wait()

    #     return out

    def _build_tree (self):
        dtype = cfg.PRECISION.np_float
        cp_dtype = cfg.PRECISION.cp_float
        float4 = cfg.PRECISION.float4  

        vertices = self._triangles.rescale(q.m).magnitude
        bounds = self._bounds.rescale(q.m).magnitude
        normals = self._normals.rescale(q.m).magnitude

        nb_vertices = len(vertices)
        nb_keys = nb_vertices // 3
        sceneMin = np.array([bounds[0], bounds[2], bounds[4], 0], dtype=dtype)
        sceneMax = np.array([bounds[1], bounds[3], bounds[5], 0], dtype=dtype)
        vertices = np.hstack((vertices, np.zeros((nb_vertices, 1))))
        vertices = cp.array(vertices, dtype=cp_dtype)
        normals = np.hstack((normals, np.zeros((len(normals), 1))))
        normals = cp.array(normals, dtype=cp_dtype)
        keys = cp.zeros(nb_keys, dtype=cp.uint32)
        rope = cp.ones(2 * nb_keys, dtype=cp.int32) * -1
        left = cp.ones(2 * nb_keys, dtype=cp.int32) * -1
        entered = cp.ones(2 * nb_keys, dtype=cp.int32) * -1
        bbMin = cp.zeros((2 * nb_keys, 4), dtype=cp_dtype)
        bbMax = cp.zeros((2 * nb_keys, 4), dtype=cp_dtype)

        # Project the triangle centroids
        args = (nb_keys, vertices, keys, bbMin, bbMax, sceneMin.view(float4), sceneMax.view(float4))
        block_size = (256, 1, 1)
        grid_size = ((nb_keys + block_size[0] - 1) // block_size[0], 1, 1)
        cfg.CUDA_TIMER.start()
        cfg.CUDA_KERNELS["project_keys"](grid_size, block_size, args)
        cp.cuda.Stream.null.synchronize()
        cfg.CUDA_TIMER.stop()
        project_t = cfg.CUDA_TIMER.elapsedTime()
        cfg.CUDA_TIMER.reset()

        # Sort the keys
        cfg.CUDA_TIMER.start()
        sorted_keys = cp.argsort(keys)
        cfg.CUDA_TIMER.stop()
        sort_t = cfg.CUDA_TIMER.elapsedTime()
        cfg.CUDA_TIMER.reset()

        keys = keys[sorted_keys]
        permutation = sorted_keys

        # Grow the tree
        args = (nb_keys, keys, permutation, rope, left, entered, bbMin, bbMax)
        block_size = (256, 1, 1)
        grid_size = ((nb_keys + block_size[0] - 1) // block_size[0], 1, 1)
        cfg.CUDA_TIMER.start()
        cfg.CUDA_KERNELS["build_tree"](grid_size, block_size, args)
        cp.cuda.Stream.null.synchronize()
        cfg.CUDA_TIMER.stop()
        grow_t = cfg.CUDA_TIMER.elapsedTime()
        cfg.CUDA_TIMER.reset()

        print (f"{nb_keys}, {project_t}, {sort_t}, {grow_t}, ", end="")

        tree = {
            "keys": keys,
            "rope": rope,
            "left": left,
            "indices": permutation,
            "bbMin": bbMin,
            "bbMax": bbMax,
            "vertices": vertices,
            "normals": normals
        }
        # t = project_t, sort_t, grow_t,

        self._tree = tree
        

    def _project(self, camera, parallel=False):
        """Projection implementation."""
        cp_dtype = cfg.PRECISION.cp_float
        float4 = cfg.PRECISION.float4
        float2 = cfg.PRECISION.float2
        uint2 = cfg.PRECISION.uint2

        U, V, W = camera.viewport_basis_vectors
        nb_keys = len(self._triangles) // 3
        image = cp.zeros(camera.shape[0] * camera.shape[1], dtype=cp_dtype)
        
        block_size = (16, 16, 1)
        grid_size = ((camera.shape[0] + block_size[0] - 1) // block_size[0], (camera.shape[1] + block_size[1] - 1) // block_size[1], 1)

        cfg.CUDA_TIMER.start()
        if parallel:
            args = (
                nb_keys, image, camera.shape.view(uint2),
                U.view(float4), V.view(float4), W.view(float4),
                camera.p00_center.view(float4), camera.pixel_size.view(float2),
                self._tree["rope"], self._tree["left"], self._tree["indices"],
                self._tree["bbMin"], self._tree["bbMax"],
                self._tree["vertices"].view(float4), self._tree["normals"].view(float4)
            )
            cfg.CUDA_KERNELS["project_parallel"](grid_size, block_size, args)
        else:
            args = (
                nb_keys, image, camera.shape.view(uint2),
                U.view(float4), V.view(float4), W.view(float4),
                camera.p00_center.view(float4), camera.ray_origin.view(float4), camera.pixel_size.view(float2),
                self._tree["rope"], self._tree["left"], self._tree["indices"],
                self._tree["bbMin"], self._tree["bbMax"],
                self._tree["vertices"].view(float4), self._tree["normals"].view(float4)
            )
            cfg.CUDA_KERNELS["project_perspective"](grid_size, block_size, args)

        cp.cuda.Stream.null.synchronize()
        cfg.CUDA_TIMER.stop()
        projectPlaneRays_t = cfg.CUDA_TIMER.elapsedTime()
        cfg.CUDA_TIMER.reset()
        image = image.get().reshape(camera.shape)

        print (f"{projectPlaneRays_t}")

        return image



    def compute_slices(self, shape, pixel_size, queue=None, out=None, offset=None):
        """Compute slices with *shape* as (z, y, x), *pixel_size*. Use *queue* and *out* for
        outuput. Offset is the starting point offset as (x, y, z).
        """
        if queue is None:
            queue = cfg.OPENCL.queue
        if out is None:
            out = cl_array.zeros(queue, shape, dtype=np.uint8)

        pixel_size = make_tuple(pixel_size, num_dims=2)
        v_1, v_2, v_3 = self._make_inputs(queue, pixel_size)
        psm = pixel_size.simplified.magnitude
        max_dx = self.max_triangle_x_diff.simplified.magnitude / psm[1]
        if offset is None:
            offset = gutil.make_vfloat3(0, 0, 0)
        else:
            offset = offset.simplified.magnitude
            offset = gutil.make_vfloat3(offset[0] / psm[1], offset[1] / psm[0], offset[2] / psm[1])

        cfg.OPENCL.programs["mesh"].compute_slices(
            queue,
            (shape[2], shape[0]),
            None,
            v_1.data,
            v_2.data,
            v_3.data,
            out.data,
            np.int32(shape[1]),
            np.int32(self.num_triangles),
            offset,
            cfg.PRECISION.np_float(max_dx),
        )

        return out

class MeshReaderDelegate(object):
    def __init__(self):
        self.dimensions = 3
        self.polydata = None
        self.bounds = None # The bounding box
        self.size = None # length of the bounding box
        self.vertices = None # Contiguous points for all triangles
        self.normals = None # Face normals
        self.triangles = None # The triangles are stored as a 1D array

class WavefrontAnimationReader(MeshReaderDelegate):
    def __init__(self, folder : str, start : int, end : int, unit: q.Quantity = q.m):
        super().__init__()
        self.folder = folder
        self.start = start
        self.end = end

        filenames = sorted (os.listdir(folder))
        self.filenames = list(islice((os.path.join(folder, f) for f in filenames if f.endswith('.obj')), start, end))
    
        self.bounds = None
        self.triangles = None
        self.vertices = None
        self.normals = None
        self.unit = unit

    @staticmethod
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
     

    def read_file (self, filename):
        self.reader = vtk.vtkOBJReader()
        self.reader.SetFileName(filename)
        self.reader.Update()

        polydata = self.reader.GetOutput()

        self.vertices, self.triangles = self.extractContiguousTriangles(polydata)
        self.bounds = polydata.GetBounds()
        self.normals = polydata.GetCellData().GetNormals()
        if self.normals is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputeCellNormalsOn()
            normals.ComputePointNormalsOff()
            normals.Update()
            self.normals = normals.GetOutput().GetCellData().GetNormals()
        
        self.bounds = np.array(self.bounds).astype(np.float32) * self.unit
        self.normals = vtk_to_numpy(self.normals).reshape(-1, 3).astype(np.float32) * self.unit
        self.vertices = self.vertices.astype(np.float32) * self.unit
        self.triangles = self.triangles.astype(np.uint32)
        ret = (self.vertices, self.normals, self.bounds)
        return ret
    
    def yield_next_timestep (self):
        for filename in self.filenames:
            yield self.read_file(filename)

class PyvistaReader(MeshReaderDelegate):
    def __init__(self, filename: str, unit: q.Quantity = q.m):
        super().__init__()
        self._filename = filename
        self._unit = unit
        self._load_mesh()

    def _load_mesh(self):
        mesh = pv.read(self._filename)
        self.polydata = mesh

        # Ensure the normals are calculated
        if mesh.cell_normals is None:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        points = mesh.points
        triangle_vertices = points[triangles]
        triangle_vertices = triangle_vertices.flatten().reshape(-1, 3)

        self.vertices = np.array(triangle_vertices).astype(np.float32) * self._unit
        self.triangles = triangles
        self.normals = np.array(mesh.cell_normals).astype(np.float32) * self._unit
        self.bounds = np.array(mesh.bounds).astype(np.float32) * self._unit

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        self._load_mesh()

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value
        self.vertices = self.vertices.rescale(value)
        self.normals = self.normals.rescale(value)
        self.bounds = self.bounds.rescale(value)

    @property
    def scene(self):
        return [self.vertices, self.normals, self.bounds]

    def visualize(self, plotter=None):
        if plotter is None:
            plotter = pv.Plotter(window_size=[800, 1024])
        plotter.add_mesh(self.polydata)
        plotter.show()

class MeshReader():
    def __init__(self, meshReader: MeshReaderDelegate):
        self._delegate = meshReader
    
    def __getattr__(self, name):
        return getattr(self._delegate, name)
    
def _extract_object(txt):
    """Extract an object from string *txt*."""
    face_start = txt.index("s ")
    if "v" not in txt[face_start:]:
        obj_end = None
    else:
        obj_end = face_start + txt[face_start:].index("v")
    subtxt = txt[:obj_end]

    pattern = r"{} (?P<x>.*) (?P<y>.*) (?P<z>.*)"
    v_pattern = re.compile(pattern.format("v"))
    f_pattern = re.compile(pattern.format("f"))
    vertices = np.array(re.findall(v_pattern, subtxt)).astype(np.float32)
    faces = np.array(re.findall(f_pattern, subtxt)).astype(np.int32).flatten() - 1

    remainder = txt[obj_end:] if obj_end else None

    return remainder, vertices, faces


def read_blender_obj(filename, objects=None):
    """Read blender wavefront *filename*, extract only *objects* which are object indices."""
    remainder = open(filename, "r").read()
    triangles = None
    face_start = 0
    i = 0

    while remainder:
        remainder, v, f = _extract_object(remainder)
        if objects is None or i in objects:
            if triangles is None:
                triangles = v[f - face_start].transpose()
            else:
                triangles = np.concatenate((triangles, v[f - face_start].transpose()), axis=1)
        face_start += len(v)
        i += 1

    return triangles


def make_cube():
    """Create a cube triangle mesh from -1 to 1 m in all dimensions."""
    seed = (-1, 1)
    points = list(itertools.product(seed, seed, seed))
    points = np.array(list(zip(*points))).reshape(3, 8)
    indices = [0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7]
    triangles = points[:, indices]
    for i in range(1, 3):
        shifted = np.roll(points, i, axis=0)[:, indices]
        triangles = np.concatenate((triangles, shifted), axis=1)

    return triangles * q.m
