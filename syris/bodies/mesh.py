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
from syris.coordinate_systems import CoordinateSystem
from syris.transformations import Transformation
from syris.bodies.base import MovableBody
from syris.devices.cameras import Camera
from syris.util import get_magnitude, make_tuple
import cupy as cp
import pyvista as pv
import os, vtk
from vtk.util.numpy_support import vtk_to_numpy
from itertools import islice
from syris.bodies.meshreader import MeshReader, PyvistaReader


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
        filename,
        trajectory,
        iterations=1,
        center="bbox",
        unit=None,
        scale=1,
        material=None,
        coordinate_system=None,
        triangles=None,
        normals=None,
        bounds=None,
    ):
        """Constructor."""
        if unit is None:
            self.unit = q.m
        else:
            self.unit = unit

        try:
            reader = MeshReader(PyvistaReader(filename, unit))
        except Exception as e:
            raise e
        
        self.mesh_name = filename

        if center is None:
            point = (0, 0, 0) * unit
        elif center == "bbox":
            point = reader.center
        elif center == "gravity":
            point = reader.center_of_mass
        elif isinstance(center, tuple):
            point = center
        else:
            self._center = None
            raise ValueError("Invalid center value")

        self._center = point * unit

        print (f"Center: {self._center}")
        
        # Create local coordinate system
        if coordinate_system is None:
            self.coordinate_system = CoordinateSystem(origin=self._center)
        else:
            self.coordinate_system = coordinate_system

        # self.coordinate_system.add_normals(normals)
        # self.coordinate_system.add_bounds(bounds)

        if triangles is None:
            self._triangles = reader.vertices.rescale(q.m).magnitude
        else:
            self._triangles = triangles
        if normals is None:
            self._normals = reader.normals.magnitude
        else:
            self._normals = normals
        if bounds is None:
            self._bounds = reader.bounds.rescale(q.m).magnitude
        else:
            self._bounds = bounds
            
        self._furthest_point = np.max(np.sqrt(np.sum(self._triangles ** 2, axis=0)))
        self._iterations = iterations

        self._current = np.copy(self._triangles)

        self._tree = None

        # Build the tree
        self._build_tree()
        self.coordinate_system.add_points(self._triangles, label=self.mesh_name)

        super(Mesh, self).__init__(trajectory, material=material)

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
        return len(self._triangles) // 3

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

    def _build_tree (self):
        dtype = cfg.PRECISION.np_float
        cp_dtype = cfg.PRECISION.cp_float
        float4 = cfg.PRECISION.float4  

        vertices = self._triangles
        bounds = self._bounds
        normals = self._normals

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

        cfg.CUDA_PIPELINE.synchronize()

        # Project the triangle centroids
        args = (nb_keys, vertices, keys, bbMin, bbMax, sceneMin.view(float4), sceneMax.view(float4))
        block_size = (256, 1, 1)
        grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
        project_t, _ = cfg.CUDA_PIPELINE.launchKernel("projectTriangleCentroid", grid_size, block_size, args)

        # Sort the keys
        sorted_keys = cp.argsort(keys)
        keys = keys[sorted_keys]
        permutation = sorted_keys

        # Grow the tree
        args = (nb_keys, keys, permutation, rope, left, entered, bbMin, bbMax)
        block_size = (256, 1, 1)
        grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
        grow_t, _ = cfg.CUDA_PIPELINE.launchKernel("growTreeKernel", grid_size, block_size, args)

        # print (f"{nb_keys}, {project_t}, {sort_t}, {grow_t}, ", end="")

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

        self._tree = tree
        

    def _project(self, camera, parallel=True, t=None, block=False):
        """Projection implementation."""
        cp_dtype = cfg.PRECISION.cp_float
        float4 = cfg.PRECISION.float4
        float2 = cfg.PRECISION.float2
        uint2 = cfg.PRECISION.uint2

        U, V, W = camera.viewport_basis_vectors
        nb_keys = len(self._triangles) // 3
        image = cp.zeros(camera.shape[0] * camera.shape[1], dtype=cp_dtype)
        
        block_size = (1, 1, 1)
        # grid_size = (1, 1, 1)
        grid_size = (int(np.ceil(camera.shape[0] * camera.shape[1] / block_size[0])), 1, 1)

        if parallel:
            args = (
                nb_keys, image, camera.shape.view(uint2),
                U.view(float4), V.view(float4), W.view(float4),
                camera.p00_center.view(float4), camera.pixel_size.view(float2),
                self._tree["rope"], self._tree["left"], self._tree["indices"],
                self._tree["bbMin"], self._tree["bbMax"],
                self._tree["vertices"].view(float4), self._tree["normals"].view(float4)
            )

            t, _ = cfg.CUDA_PIPELINE.launchKernel("projectParallelKernel", grid_size, block_size, args)

        else:
            args = (
                nb_keys, image, camera.shape.view(uint2),
                U.view(float4), V.view(float4), W.view(float4),
                camera.p00_center.view(float4), camera.ray_origin.view(float4), camera.pixel_size.view(float2),
                self._tree["rope"], self._tree["left"], self._tree["indices"],
                self._tree["bbMin"], self._tree["bbMax"],
                self._tree["vertices"].view(float4), self._tree["normals"].view(float4)
            )

             
            t, _ = cfg.CUDA_PIPELINE.launchKernel("projectParallelKernel", grid_size, block_size, args)

        # print (f"{t}")

        return image.reshape(camera.shape).get()

    def visualize_bvh(self, plotter, mapto="id", cmap="Pastel1"):
        """Visualize the bounding volume hierarchy."""
        if self._tree is None:
            raise ValueError("Tree is not built yet")
        
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        
        cmapper = plt.get_cmap(cmap)

        vertices = self._tree["vertices"]
        bbMin = self._tree["bbMin"]
        bbMax = self._tree["bbMax"]
        nb_keys = len(self._triangles) // 3

        # Get the bounding boxes data
        bbMin_np = bbMin.get()
        bbMax_np = bbMax.get()

        if mapto == "id":
            minv, maxv = 0, nb_keys
            values = np.arange(minv, maxv)
        elif mapto == "depth":
            minv, maxv = 0, np.log2(nb_keys)
            values = np.log2(np.arange(minv, maxv))
        elif mapto == "volume":
            # use np to calculate the volume of each bounding box
            diff = bbMax_np - bbMin_np
            volumes = np.prod(diff, axis=1)
            minv, maxv = np.min(volumes), np.max(volumes)
            values = volumes
        else:
            raise ValueError("Invalid mapto value")
        
        # Normalize the values
        norm = mcolors.Normalize(vmin=minv, vmax=maxv)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # Visualize the leaf nodes
        for i in range(nb_keys, 2 * nb_keys):
            idx = i - nb_keys  # Index in the volumes array
            bounds = (
                bbMin_np[i, 0],  # xMin
                bbMax_np[i, 0],  # xMax
                bbMin_np[i, 1],  # yMin
                bbMax_np[i, 1],  # yMax
                bbMin_np[i, 2],  # zMin
                bbMax_np[i, 2],  # zMax
            )
            plotter.add_mesh(pv.Box(bounds=bounds), color=sm.to_rgba(values[idx]), opacity=0.5)

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
