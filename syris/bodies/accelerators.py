import abc
import syris.config as cfg
import quantities as q
from syris.util import get_magnitude, make_tuple
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import syris.gpu.util as gutil
import numpy as np
import logging

LOG = logging.getLogger(__name__)

class AcceleratorBase(abc.ABC):
    def __init__(self, mesh):
        self.mesh = mesh
        self.backend = cfg.BACKEND
        self._built_for_state = -1
        
    @abc.abstractmethod
    def build(self, **kwargs):
        """
        Prepares the acceleration structure. Accepts backend-specific arguments.
        """
        pass

    @abc.abstractmethod
    def project(self, shape, pixel_size, offset, /, *, t=None, **kwargs):
        pass

class BvhCupyAccelerator(AcceleratorBase):
    """The new, fast GPU-based projection strategy using a BVH tree and CUDA."""
    def __init__(self, mesh):
        super().__init__(mesh)
        self.pipeline = cfg.BACKEND.pipeline
        if self.pipeline is None:
            raise RuntimeError("CUDA backend is active, but the CudaPipeline was not initialized.")
        self._tree = None

    def build(self, **kwargs):
        """
        Build a bounding volume hierarchy tree for the mesh.
        """

        self.mesh.transform()

        xp = cfg.BACKEND.xp
        
        float = cfg.PRECISION.np_float
        float4 = cfg.PRECISION.float4

        host_vertices = self.mesh._current[0:3, :]
        nb_vertices = host_vertices.shape[1]
        bounds = self.mesh.bounds

        if self.mesh._use_normals:
            host_normals = self.mesh.normals
        else:
            host_normals = None

        nb_keys = nb_vertices // 3
        sceneMin = np.array([bounds[0], bounds[2], bounds[4], 0], dtype=float)
        sceneMax = np.array([bounds[1], bounds[3], bounds[5], 0], dtype=float)

        t_epsilon = self.mesh.epsilon
        
        try:
            host_vertices = host_vertices.T            
            if len(host_vertices.shape) == 1:
                host_vertices = host_vertices.reshape(-1, 3)
                nb_vertices = len(host_vertices)

            # Ensure vertices have the correct shape before proceeding
            if len(host_vertices.shape) != 2 or host_vertices.shape[1] != 3:
                raise ValueError(f"Vertices must have shape (N, 3), but got {host_vertices.shape}")

            # Create and transfer vertex data to the GPU in a single operation
            vertices = xp.zeros((nb_vertices, 4), dtype=float)
            vertices[:, :3] = xp.array(host_vertices, dtype=float)
            
            # Create and transfer normal data to the GPU if it exists
            if host_normals is not None:
                normals = xp.zeros((nb_keys, 4), dtype=float)
                normals[:, :3] = xp.array(host_normals, dtype=float)
            else:
                normals = None

            keys = xp.zeros(nb_keys, dtype=xp.uint64)
            rope = xp.ones(2 * nb_keys, dtype=xp.int32) * -1
            left = xp.ones(2 * nb_keys, dtype=xp.int32) * -1
            entered = xp.ones(2 * nb_keys, dtype=xp.int32) * -1
            bbMin = xp.zeros((2 * nb_keys, 4), dtype=float)
            bbMax = xp.zeros((2 * nb_keys, 4), dtype=float)

            # Project the triangle centroids
            args = (nb_keys, vertices, keys, bbMin.view(float4), bbMax.view(float4), sceneMin.view(float4), sceneMax.view(float4))
            block_size = (256, 1, 1)
            grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
            project_t = cfg.BACKEND.pipeline.launchKernel("projectTriangleCentroid", grid_size, block_size, args)
            LOG.debug(f"Projected triangle centroids in {project_t} ms")

            # Sort the keys
            sorted_keys = xp.argsort(keys)
            keys = keys[sorted_keys]
            permutation = sorted_keys

            # Grow the tree
            args = (nb_keys, keys, permutation, rope, left, entered, bbMin.view(float4), bbMax.view(float4))
            block_size = (256, 1, 1)
            grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
            grow_t = cfg.BACKEND.pipeline.launchKernel("growTreeKernel", grid_size, block_size, args)
            LOG.debug(f"grew tree in {grow_t} ms")

            # Free memory we no longer need
            cfg.BACKEND.pipeline.synchronize()

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
                "t_epsilon": t_epsilon
            }

            self._tree = tree
            self._built_for_state = self.mesh._state

            
        except Exception as e:
            self._built_for_state = -1

            # Clean up CUDA memory in case of error
            cfg.BACKEND.pipeline.synchronize()
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
            raise RuntimeError(f"Failed to build BVH tree: {e}") from e



    def project(self, shape, pixel_size, offset, /, *, t=None, iterations=0, abs_tolerance = 1e-5, rel_tolerance = .02, **kwargs):
        """Projection implementation using cuda ray caster."""

        if self._tree is None:
            raise RuntimeError("BVH tree must be built before projection.")
        
        camera = kwargs.get('camera')
        parallel = kwargs.get('parallel', True)

        if self._tree == None:
            raise RuntimeError("Tree must be built first")

        xp = cfg.BACKEND.xp

        float = cfg.PRECISION.np_float
        float4 = cfg.PRECISION.float4
        float2 = cfg.PRECISION.float2
        uint2 = cfg.PRECISION.uint2

        nb_keys = self._tree["keys"].shape[0]
        use_normals = (self._tree["normals"] is not None)
        U, V, W = camera.viewport_basis_vectors
        image = xp.ones(camera.shape[0] * camera.shape[1], dtype=float) * -1

        global_counter = xp.zeros(1, dtype=xp.uint32)
        
        block_size = (16, 1, 1)
        grid_size = (int(xp.ceil(camera.shape[0] * camera.shape[1] / block_size[0])), 1, 1)

        base_args = [
            global_counter, nb_keys, image,
            self._tree["vertices"].view(float4)
        ]

        camera_args = [
            camera.shape.view(uint2), U.view(float4), V.view(float4), W.view(float4),
            camera.p00_corner.view(float4), camera.pixel_size.view(float2),
        ]

        tree_args = [
            self._tree["rope"], self._tree["left"], self._tree["indices"],
            self._tree["bbMin"], self._tree["bbMax"],
            self._tree["sceneMin"].view(float4), self._tree["sceneMax"].view(float4)
        ]

        sampling_args = [
            float(abs_tolerance),
            float(rel_tolerance),
            int(iterations)
        ]

        epsilon_args = [
            float(5.0 * (2**-24)),  # p_ray_box_epsilon
            float(1e-7),            # p_tri_ray_tmin
            float(256.0),           # p_tri_gamma_multiplier
            float(1e-10),           # p_tri_abs_min_error
            float(128.0),           # p_tri_d_gamma_multiplier (passed as float, cast in kernel)
            float(1e-100),          # p_tri_d_abs_min_error (passed as float, cast in kernel)
            float(1e-6),            # p_group_abs_epsilon
            float(1e-5),            # p_group_rel_epsilon
            float(1e-7)             # p_unique_abs_epsilon (for no-normal traceRay)
        ]

        if not parallel:
            camera_args.append(camera.source_point.view(float4))

        if use_normals:
            base_args.append(self._tree["normals"].view(float4))

        if parallel and use_normals:
            kernel_name = "project_parallel_normals_kernel"
        elif parallel and not use_normals:
            kernel_name = "project_parallel_kernel"
        elif not parallel and use_normals:
            kernel_name = "project_conebeam_normals_kernel"
        else:
            kernel_name = "project_conebeam_kernel"

        args = []
        args.extend(base_args)
        args.extend(camera_args)
        args.extend(tree_args)
        args.extend(sampling_args)
        args.extend(epsilon_args)

        LOG.debug(f"Launching kernel: {kernel_name}")
        args = tuple(args)
        time, _ = cfg.BACKEND.pipeline.launchKernel(kernel_name, grid_size, block_size, args)
        LOG.debug(f"Projected mesh in {time} ms")

        cfg.BACKEND.pipeline.synchronize()

        img = image.reshape(camera.shape).get()

        import matplotlib.pyplot as plt
        plt.imshow(img)

        return img
    
    # TODO: fix params for this method
    # def _get_color_mapping_values(self, mapto, nb_keys, bbMin_np, bbMax_np):
    #     xp = cfg.BACKEND.xp

    #     """Helper to calculate the values for color mapping based on the chosen attribute."""
    #     if mapto == "id":
    #         return xp.arange(nb_keys, dtype=float)

    #     elif mapto == "depth":
    #         # This section still relies on placeholder data.
    #         # For production, this should be replaced with actual depth calculation.
    #         if hasattr(self, '_leaf_depths_example') and len(self._leaf_depths_example) == nb_keys:
    #             return xp.array(self._leaf_depths_example, dtype=float)
    #         else:
    #             log.warning("Using dummy depth data for 'mapto=depth'.")
    #             if nb_keys == 0:
    #                 return xp.array([], dtype=float)
    #             base_depth = xp.log2(nb_keys) if nb_keys > 1 else 1.0
    #             depths = xp.random.uniform(base_depth * 0.8, base_depth * 1.5, size=nb_keys)
    #             return depths if nb_keys > 1 else xp.array([1.0])

    #     elif mapto == "volume":
    #         if bbMin_np.shape[0] < 2 * nb_keys:
    #             raise ValueError("Bounding box arrays are too small for leaf node indexing.")
            
    #         leaf_bbMin = bbMin_np[nb_keys : 2 * nb_keys, :3]
    #         leaf_bbMax = bbMax_np[nb_keys : 2 * nb_keys, :3]
            
    #         # Calculate volumes, adding a small epsilon to avoid log(0)
    #         volumes_raw = xp.prod(leaf_bbMax - leaf_bbMin, axis=1)
    #         return xp.log(volumes_raw + 1e-9)
            
    #     else:
    #         raise ValueError(f"Invalid 'mapto' value: {mapto}")

    # def visualize_bvh(self, plotter, mapto="id", cmap="viridis"):
    #     """
    #     Visualize the bounding volume hierarchy by coloring leaf nodes.

    #     Args:
    #         plotter: A pyvista.Plotter object to add the meshes to.
    #         mapto (str): The attribute to map to colors. One of 'id', 'depth', or 'volume'.
    #         cmap (str): The name of the matplotlib colormap to use.
    #     """
    #     xp = cfg.BACKEND.xp

    #     if self._tree is None:
    #         raise ValueError("BVH tree has not been built yet.")

    #     nb_keys = len(self._triangles) // 3
    #     if nb_keys == 0:
    #         log.debug("No BVH keys to visualize.")
    #         return

    #     bbMin_np = self._tree["bbMin"].get()  # Assuming .get() returns a numpy array
    #     bbMax_np = self._tree["bbMax"].get()

    #     # 1. Get the values for color mapping
    #     values = self._get_color_mapping_values(mapto, nb_keys, bbMin_np, bbMax_np)

    #     if values.size == 0:
    #         log.debug("No values to visualize.")
    #         return

    #     # 2. Normalize the values for the colormap
    #     minv, maxv = xp.min(values), xp.max(values)
    #     if minv == maxv:
    #         # Avoid division by zero if all values are the same
    #         norm = mcolors.Normalize(vmin=minv - 1e-6, vmax=maxv + 1e-6)
    #     else:
    #         norm = mcolors.Normalize(vmin=minv, vmax=maxv)
        
    #     scalar_mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)

    #     # 3. Create and add meshes for the leaf nodes
    #     for i in range(nb_keys):
    #         tree_index = i + nb_keys
            
    #         bounds = (
    #             bbMin_np[tree_index, 0], bbMax_np[tree_index, 0],  # xMin, xMax
    #             bbMin_np[tree_index, 1], bbMax_np[tree_index, 1],  # yMin, yMax
    #             bbMin_np[tree_index, 2], bbMax_np[tree_index, 2],  # zMin, zMax
    #         )

    #         # Check for invalid bounds (min > max)
    #         if not (bounds[1] >= bounds[0] and bounds[3] >= bounds[2] and bounds[5] >= bounds[4]):
    #             log.warning(f"Skipping box {i} due to invalid bounds: {bounds}")
    #             continue

    #         color = scalar_mappable.to_rgba(values[i])
    #         box_mesh = pv.Box(bounds=bounds)
    #         plotter.add_mesh(box_mesh, color=color, show_edges=True, opacity=0.9)


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
        

    def project(self, shape, pixel_size, offset, /, *, t=None, **kwargs):
        """Projection implementation."""
        xp = cfg.BACKEND.xp
        queue = kwargs.get('queue', cfg.OPENCL.queue)
        out = kwargs.get('out')
        block = kwargs.get('block', False)

        def get_crop(index, fov):
            minimum = max(self.mesh.extrema[index][0], fov[index][0])
            maximum = min(self.mesh.extrema[index][1], fov[index][1])

            return minimum - offset[::-1][index], maximum - offset[::-1][index]

        def get_px_value(value, round_func, ps):
            return int(round_func(get_magnitude(value / ps)))

        psm = pixel_size.simplified.magnitude
        fov = offset + shape * pixel_size
        fov = (
            xp.concatenate((offset.simplified.magnitude[::-1], fov.simplified.magnitude[::-1]))
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
            offset = gutil.make_vfloat2(*(offset / pixel_size).simplified.magnitude[::-1])

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
            raise RuntimeError("CUDA backend is active, but the CudaPipeline was not initialized.")
        self._tree = None
    
    def build(self, **kwargs):
        """
        Prepares the mesh for projection by applying transformations and sorting.
        This version takes no arguments as it operates on the stored mesh.
        """
        self.mesh.transform()
        self.mesh.sort()

    def project(self, shape, pixel_size, offset, /, *, t=None, **kwargs):
        """Projection implementation."""
        xp = cfg.BACKEND.xp

        camera = kwargs.get('camera')

        
        block_size = (1, 1, 1)
        grid_size = (shape[0], shape[1], 1)

        float = cfg.PRECISION.np_float
        uint2 = cfg.PRECISION.uint2
        float2 = cfg.PRECISION.float2
        float3 = cfg.PRECISION.float3

        psm = pixel_size
        pixel_size = pixel_size.rescale(cfg.UNIT)
        psm_np = np.array([psm[1], psm[0], psm[1]], dtype=float)
        def make_scaled_vertices(vertex_index, px_size):
            verts = self.mesh._current[:-1, vertex_index::3] / px_size.rescale(cfg.UNIT).magnitude
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
            np.concatenate((offset.rescale(cfg.UNIT).magnitude[::-1], fov.rescale(cfg.UNIT).magnitude[::-1]))
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
            block_size = (1, 1, 1) # Use a more efficient block size
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
                V1.view(float3), V2.view(float3), V3.view(float3), 
                int32(nb_triangles), 
                image, 
                int32(full_image_width),
                kernel_offset.view(uint2),
                kernel_mesh_offset.view(float2), 
                float32(psm[1]),
                float32(max_dx), 
                float32(min_z), 
                int32(self.mesh.iterations)
            ]

            args = tuple(args)
            time, _ = cfg.BACKEND.pipeline.launchKernel("compute_thickness_kernel", grid_size, block_size, args)

            xp.cuda.Stream.null.synchronize()

        return image.reshape(shape)