import abc
import syris.config as cfg
import quantities as q
from syris.util import get_magnitude
import pyopencl.cltypes as cltypes
import pyopencl.array as cl_array
import syris.gpu.util as gutil
import numpy as np
import logging

LOG = logging.getLogger(__name__)


class AcceleratorBase(abc.ABC):
    def __init__(self, mesh):
        self.mesh = mesh
        self.backend = cfg.BACKEND
        self._built_for_state = -1
        self.tree = None

    @abc.abstractmethod
    def build(self, **kwargs):
        """
        Prepares the acceleration structure. Accepts backend-specific arguments.
        """
        pass

    @abc.abstractmethod
    def project(self, shape=None, pixel_size=None, /, **kwargs):
        pass


class BvhCupyAccelerator(AcceleratorBase):
    """CUDA-based ray caster accelerated by a LBVH tree"""

    def __init__(self, mesh):
        super().__init__(mesh)
        self.pipeline = cfg.BACKEND.pipeline
        if self.pipeline is None:
            raise RuntimeError(
                "CUDA backend is active, but the CudaPipeline was not initialized."
            )

        self.xp = cfg.BACKEND.xp

        self.float_dtype = cfg.PRECISION.np_float
        self.float4_view = cfg.PRECISION.float4
        self.float2_view = cfg.PRECISION.float2
        self.uint2_view = cfg.PRECISION.uint2

        is_double = self.float_dtype is np.float64
        rc_cfg = cfg.RayCasting(is_double, self.mesh.dynamic_range)

        self._epsilon_args = (
            self.float_dtype(rc_cfg.p_ray_box_epsilon),
            self.float_dtype(rc_cfg.p_tri_ray_tmin),
            self.float_dtype(rc_cfg.p_tri_gamma_multiplier),
            self.float_dtype(rc_cfg.p_tri_abs_min_error),
            self.float_dtype(rc_cfg.p_tri_d_gamma_multiplier),
            self.float_dtype(rc_cfg.p_tri_d_abs_min_error),
            self.float_dtype(rc_cfg.p_group_abs_epsilon),
            self.float_dtype(rc_cfg.p_group_rel_epsilon),
            self.float_dtype(rc_cfg.p_unique_abs_epsilon),
        )

        self.tree = None

    def build(self, **kwargs):
        """
        Build a bounding volume hierarchy tree for the mesh.
        """
        self.mesh.transform()

        host_vertices = self.mesh._current[0:3, :]
        nb_vertices = host_vertices.shape[1]
        nb_keys = self.mesh.num_triangles

        host_normals = self.mesh.normals if self.mesh._use_normals else None

        bounds = self.mesh.bounds
        sceneMin = np.array(
            [bounds[0], bounds[2], bounds[4], 0], dtype=self.float_dtype
        )
        sceneMax = np.array(
            [bounds[1], bounds[3], bounds[5], 0], dtype=self.float_dtype
        )
        t_epsilon = self.mesh.epsilon

        host_vertices = host_vertices.T
        if len(host_vertices.shape) == 1:
            host_vertices = host_vertices.reshape(-1, 3)
            nb_vertices = len(host_vertices)

        if len(host_vertices.shape) != 2 or host_vertices.shape[1] != 3:
            raise ValueError(
                f"Vertices must have shape (N, 3), but got {host_vertices.shape}"
            )

        try:
            # Create and transfer vertex data to the GPU in a single operation
            vertices = self.xp.zeros((nb_vertices, 4), dtype=self.float_dtype)
            vertices[:, :3] = self.xp.array(host_vertices, dtype=self.float_dtype)

            # Create and transfer normal data to the GPU if it exists
            normals = None
            if host_normals is not None:
                normals = self.xp.zeros((nb_keys, 4), dtype=self.float_dtype)
                normals[:, :3] = self.xp.array(host_normals, dtype=self.float_dtype)

            keys = self.xp.zeros(nb_keys, dtype=self.xp.uint64)
            rope = self.xp.full(2 * nb_keys, -1, dtype=self.xp.int32)
            left = self.xp.full(2 * nb_keys, -1, dtype=self.xp.int32)
            entered = self.xp.full(2 * nb_keys, -1, dtype=self.xp.int32)
            bbMin = self.xp.zeros((2 * nb_keys, 4), dtype=self.float_dtype)
            bbMax = self.xp.zeros((2 * nb_keys, 4), dtype=self.float_dtype)

            # Project the triangle centroids
            block_size = (256, 1, 1)
            grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)

            project_args = (
                nb_keys,
                vertices,
                keys,
                bbMin.view(self.float4_view),
                bbMax.view(self.float4_view),
                sceneMin.view(self.float4_view),
                sceneMax.view(self.float4_view),
            )
            project_t = self.pipeline.launchKernel(
                "projectTriangleCentroid", grid_size, block_size, project_args
            )
            LOG.debug(f"Projected triangle centroids in {project_t} ms")

            # Sort the keys
            sorted_keys = self.xp.argsort(keys)
            keys = keys[sorted_keys]
            permutation = sorted_keys

            # Grow the tree
            grow_args = (
                nb_keys,
                keys,
                permutation,
                rope,
                left,
                entered,
                bbMin.view(self.float4_view),
                bbMax.view(self.float4_view),
            )
            block_size = (256, 1, 1)
            grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
            grow_t = cfg.BACKEND.pipeline.launchKernel(
                "growTreeKernel", grid_size, block_size, grow_args
            )
            LOG.debug(f"grew tree in {grow_t} ms")

            # Free memory we no longer need
            self.pipeline.synchronize()

            tree = {
                "keys": keys,
                "rope": rope,
                "left": left,
                "indices": permutation,
                "bbMin": bbMin,
                "bbMax": bbMax,
                "sceneMin": sceneMin,
                "sceneMax": sceneMax,
                "vertices": vertices,
                "normals": normals,
                "t_epsilon": t_epsilon,
            }

            self.tree = tree
            self._built_for_state = self.mesh._state

        except Exception as e:
            self._built_for_state = -1
            self.pipeline.synchronize()
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()
            raise RuntimeError(f"Failed to build BVH tree: {e}") from e

    def project(
        self,
        shape=None,
        pixel_size=None,
        /,
        *,
        iterations=0,
        abs_tolerance=1e-5,
        rel_tolerance=0.02,
        camera=None,
        **kwargs,
    ):
        if self.tree is None:
            raise RuntimeError("BVH tree must be built before projection.")

        if camera is None:
            raise ValueError("Missing required keyword argument: 'camera'")

        parallel = kwargs.get("parallel", True)

        nb_keys = self.tree["keys"].shape[0]
        use_normals = self.tree["normals"] is not None
        U, V, W = camera.viewport_basis_vectors

        # --- Calculate Scaling ---
        # Scale geometry to [-1, 1] for numerical stability in kernel
        max_bound = float(self.mesh.furthest_point.magnitude)
        scale = 1.0
        if max_bound > 0:
            scale = float(1.0 / max_bound)

        # --- Prepare Kernel Launch ---
        image = (
            self.xp.ones(camera.shape[0] * camera.shape[1], dtype=self.float_dtype) * -1
        )
        global_counter = self.xp.zeros(1, dtype=self.xp.uint32)

        block_size = (256, 1, 1)
        grid_size = (
            int(self.xp.ceil(camera.shape[0] * camera.shape[1] / block_size[0])),
            1,
            1,
        )

        # --- Build Kernel Arguments ---
        base_args = [
            global_counter,
            nb_keys,
            image,
            (self.tree["vertices"] * scale).view(self.float4_view),
        ]
        if use_normals:
            base_args.append((self.tree["normals"] * scale).view(self.float4_view))

        camera_args = [
            camera.kernel_shape_xy.view(self.uint2_view),
            U.view(self.float4_view),
            V.view(self.float4_view),
            W.view(self.float4_view),
            (camera.p00_corner * scale).view(self.float4_view),
            (camera.kernel_pixel_size_xy * scale).view(self.float2_view),
        ]

        if not parallel:
            camera_args.append((camera.source_point * scale).view(self.float4_view))

        tree_args = [
            self.tree["rope"],
            self.tree["left"],
            self.tree["indices"],
            self.tree["bbMin"] * scale,
            self.tree["bbMax"] * scale,
            (self.tree["sceneMin"] * scale).view(self.float4_view),
            (self.tree["sceneMax"] * scale).view(self.float4_view),
        ]

        sampling_args = [float(abs_tolerance), float(rel_tolerance), int(iterations)]

        # --- Select Kernel Name ---
        mode = "parallel" if parallel else "conebeam"
        suffix = "_normals" if use_normals else ""
        kernel_name = f"project_{mode}{suffix}_kernel"

        # --- Assemble All Arguments and Launch ---
        all_args = (
            *base_args,
            *camera_args,
            *tree_args,
            *sampling_args,
            *self._epsilon_args,
        )

        LOG.debug(f"Launching kernel: {kernel_name}")
        time, _ = self.pipeline.launchKernel(
            kernel_name, grid_size, block_size, all_args
        )
        LOG.debug(f"Projected mesh in {time} ms")

        self.pipeline.synchronize()

        img = image.reshape(camera.shape)

        # Rescale image values back to original world coordinates
        return img * max_bound


class LegacyCpuAccelerator(AcceleratorBase):
    """The fallback CPU-based projection strategy (based on OpenCL version)."""

    def __init__(self, mesh):
        super().__init__(mesh)

    def build(self, **kwargs):
        """
        Prepares the mesh for projection by applying transformations and sorting.
        This version takes no arguments as it operates on the stored mesh.
        """
        self.mesh.transform()
        self.mesh.sort()

    def project(self, shape=None, pixel_size=None, /, **kwargs):
        """Projection implementation."""
        xp = cfg.BACKEND.xp
        queue = kwargs.pop("queue", None)
        out = kwargs.pop("out", None)
        block = kwargs.pop("block", False)
        offset = kwargs.pop("offset", False)

        if queue is None:
            queue = cfg.OPENCL.queue

        if out is None:
            out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_float)

        def get_crop(index, fov):
            minimum = max(self.mesh.extrema[index][0], fov[index][0])
            maximum = min(self.mesh.extrema[index][1], fov[index][1])

            return minimum - offset[::-1][index], maximum - offset[::-1][index]

        def get_px_value(value, round_func, ps):
            return int(round_func(get_magnitude(value / ps)))

        psm = pixel_size.simplified.magnitude
        fov = offset + shape * pixel_size
        fov = (
            xp.concatenate(
                (offset.simplified.magnitude[::-1], fov.simplified.magnitude[::-1])
            )
            .reshape(2, 2)
            .transpose()
            * q.m
        )

        if (
            self.mesh.extrema[0][0] < fov[0][1]
            and self.mesh.extrema[0][1] > fov[0][0]
            and self.mesh.extrema[1][0] < fov[1][1]
            and self.mesh.extrema[1][1] > fov[1][0]
        ):
            # Object inside FOV
            x_min, x_max = get_crop(0, fov)
            y_min, y_max = get_crop(1, fov)
            x_min_px = get_px_value(x_min, xp.floor, pixel_size[1])
            x_max_px = get_px_value(x_max, xp.ceil, pixel_size[1])
            y_min_px = get_px_value(y_min, xp.floor, pixel_size[0])
            y_max_px = get_px_value(y_max, xp.ceil, pixel_size[0])
            width = min(x_max_px - x_min_px, shape[1])
            height = min(y_max_px - y_min_px, shape[0])
            compute_offset = cltypes.make_int2(x_min_px, y_min_px)
            v_1, v_2, v_3 = self.mesh._make_inputs(queue, pixel_size)
            max_dx = self.mesh.max_triangle_x_diff.simplified.magnitude / psm[1]
            # Use the same pixel size as for the x-axis, which will work for objects "not too far"
            # from the imaging plane
            min_z = self.mesh.extrema[2][0].simplified.magnitude / psm[1]
            offset = gutil.make_vfloat2(
                *(offset / pixel_size).simplified.magnitude[::-1]
            )

            ev = cfg.OPENCL.programs["mesh"].compute_thickness(
                queue,
                (width, height),
                None,
                v_1.data,
                v_2.data,
                v_3.data,
                out.data,
                xp.int32(self.mesh.num_triangles),
                xp.int32(shape[1]),
                compute_offset,
                offset,
                cfg.PRECISION.np_float(psm[1]),
                cfg.PRECISION.np_float(max_dx),
                cfg.PRECISION.np_float(min_z),
                xp.int32(self.mesh.iterations),
            )
            if block:
                ev.wait()

        return out


class LegacyCUDAAccelerator(AcceleratorBase):
    """The fallback CPU-based projection strategy (based on OpenCL version)."""

    def __init__(self, mesh):
        super().__init__(mesh)
        self.pipeline = cfg.BACKEND.pipeline
        if self.pipeline is None:
            raise RuntimeError(
                "CUDA backend is active, but the CudaPipeline was not initialized."
            )
        self.tree = None

    def build(self, **kwargs):
        """
        Prepares the mesh for projection by applying transformations and sorting.
        This version takes no arguments as it operates on the stored mesh.
        """
        self.mesh.transform()
        self.mesh.sort()

    def project(self, shape=None, pixel_size=None, /, **kwargs):
        """Projection implementation."""
        xp = cfg.BACKEND.xp

        offset = kwargs.pop("offset", (0,0))

        block_size = (1, 1, 1)
        grid_size = (shape[0], shape[1], 1)

        float = cfg.PRECISION.np_float
        uint2 = cfg.PRECISION.uint2
        float2 = cfg.PRECISION.float2
        float3 = cfg.PRECISION.float3

        psm = pixel_size
        pixel_size = pixel_size.rescale(cfg.UNIT)

        def make_scaled_vertices(vertex_index, px_size):
            verts = (
                self.mesh._current[:-1, vertex_index::3]
                / px_size.rescale(cfg.UNIT).magnitude
            )
            return verts.transpose().flatten().astype(float)

        v1_host = make_scaled_vertices(0, pixel_size[1])
        v2_host = make_scaled_vertices(1, pixel_size[0])
        v3_host = make_scaled_vertices(2, pixel_size[1])

        V1 = xp.array(v1_host)
        V2 = xp.array(v2_host)
        V3 = xp.array(v3_host)

        nb_triangles = self.mesh.num_triangles
        image = xp.zeros(shape[0] * shape[1], dtype=float)

        def get_crop(index, fov):
            minimum = max(self.mesh.extrema[index][0], fov[index][0])
            maximum = min(self.mesh.extrema[index][1], fov[index][1])

            return minimum - offset[::-1][index], maximum - offset[::-1][index]

        def get_px_value(value, round_func, ps):
            return int(round_func(get_magnitude(value / ps)))

        offset = offset.rescale(cfg.UNIT)

        fov = offset + shape * pixel_size
        fov = (
            np.concatenate(
                (
                    offset.rescale(cfg.UNIT).magnitude[::-1],
                    fov.rescale(cfg.UNIT).magnitude[::-1],
                )
            )
            .reshape(2, 2)
            .transpose()
            * q.m
        )

        max_dx = 0
        min_z = 0

        if (
            self.mesh.extrema[0][0] < fov[0][1]
            and self.mesh.extrema[0][1] > fov[0][0]
            and self.mesh.extrema[1][0] < fov[1][1]
            and self.mesh.extrema[1][1] > fov[1][0]
        ):
            # Calculate the cropped ROI dimensions
            x_min, x_max = get_crop(0, fov)
            y_min, y_max = get_crop(1, fov)
            x_min_px = get_px_value(x_min, np.floor, pixel_size[1])
            x_max_px = get_px_value(x_max, np.ceil, pixel_size[1])
            y_min_px = get_px_value(y_min, np.floor, pixel_size[0])
            y_max_px = get_px_value(y_max, np.ceil, pixel_size[0])

            # These are the actual dimensions of the work to be done
            roi_width = min(x_max_px - x_min_px, shape[1])
            roi_height = min(y_max_px - y_min_px, shape[0])

            # Set the kernel launch grid to the smaller ROI size
            block_size = (1, 1, 1)  # Use a more efficient block size
            grid_size = (roi_width, roi_height, 1)

            # Prepare parameters for the kernel
            # The kernel needs the offset to know where to write in the full image
            kernel_offset = np.array([x_min_px, y_min_px], dtype=np.int32)

            # The kernel needs the full image width for memory indexing
            full_image_width = shape[1]

            max_dx = self.mesh.max_triangle_x_diff.rescale(cfg.UNIT).magnitude / psm[1]
            min_z = self.mesh.extrema[2][0].rescale(cfg.UNIT).magnitude / psm[1]
            kernel_mesh_offset = (offset / pixel_size).magnitude[::-1]
            kernel_mesh_offset = np.array(kernel_mesh_offset, dtype=float)

            float32 = cfg.PRECISION.np_float
            int32 = np.int32

            args = [
                V1.view(float3),
                V2.view(float3),
                V3.view(float3),
                int32(nb_triangles),
                image,
                int32(full_image_width),
                kernel_offset.view(uint2),
                kernel_mesh_offset.view(float2),
                float32(psm[1]),
                float32(max_dx),
                float32(min_z),
                int32(self.mesh.iterations),
            ]

            args = tuple(args)
            time, _ = cfg.BACKEND.pipeline.launchKernel(
                "compute_thickness_kernel", grid_size, block_size, args
            )

            xp.cuda.Stream.null.synchronize()

        return image.reshape(shape)
