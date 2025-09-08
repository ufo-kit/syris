import abc
import syris.config as cfg
import quantities as q
from syris.util import get_magnitude, make_tuple
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import syris.gpu.util as gutil
import numpy as np

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
    def project(self, shape, pixel_size, /, *, t=None, offset=None, **kwargs):
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

        host_vertices = self.mesh.triangles.magnitude

        print (host_vertices)

        nb_vertices = host_vertices.shape[1]
        bounds = self.mesh.bounds
        host_normals = self.mesh.normals

        nb_keys = nb_vertices // 3
        sceneMin = np.array([bounds[0], bounds[2], bounds[4], 0], dtype=float)
        sceneMax = np.array([bounds[1], bounds[3], bounds[5], 0], dtype=float)

        t_epsilon = self.mesh.epsilon * .01

        print (f"{t_epsilon:.12f}")

        print(f"Building tree with {nb_vertices} vertices, {nb_keys} triangles")
        print(f"Vertices shape: {host_vertices.shape}")
        
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

            keys = xp.zeros(nb_keys, dtype=xp.uint32)
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

            # Sort the keys
            sorted_keys = xp.argsort(keys)
            keys = keys[sorted_keys]
            permutation = sorted_keys

            # Grow the tree
            args = (nb_keys, keys, permutation, rope, left, entered, bbMin.view(float4), bbMax.view(float4))
            block_size = (256, 1, 1)
            grid_size = (int(np.ceil(nb_keys / block_size[0])), 1, 1)
            grow_t = cfg.BACKEND.pipeline.launchKernel("growTreeKernel", grid_size, block_size, args)

            # Free memory we no longer need
            del entered  # We don't store this in the tree
            xp.cuda.Stream.null.synchronize()

            tree = {
                "keys": keys,
                "rope": rope,
                "left": left, 
                "indices": permutation,
                "bbMin": bbMin,
                "bbMax": bbMax,
                "vertices": vertices,
                "normals": normals,
                "t_epsilon": t_epsilon
            }

            self._tree = tree

            self._built_for_state = self.mesh._state

            
        except Exception as e:
            self._built_for_state = -1

            # Clean up CUDA memory in case of error
            xp.cuda.Stream.null.synchronize()
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
            print(f"Error building BVH tree: {e}")
            # Re-raise with more info
            raise RuntimeError(f"Failed to build BVH tree: {e}") from e



    def project(self, shape, pixel_size, /, *, t=None, offset=None, **kwargs):
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

        args = [
            nb_keys, image, camera.shape.view(uint2),
            U.view(float4), V.view(float4), W.view(float4),
            camera.p00_center.view(float4), camera.pixel_size.view(float2),
            self._tree["rope"], self._tree["left"], self._tree["indices"],
            self._tree["bbMin"], self._tree["bbMax"],
            self._tree["vertices"].view(float4), global_counter,
        ]

        # for arg in args:
        #     print(f'{arg=}')

        if use_normals:
            args.append(self._tree["normals"].view(float4))
        
        if not parallel:
            args.append(camera.source_point.view(float4))

        args.append(self._tree["t_epsilon"])

        if parallel and use_normals:
            kernel_name = "project_parallel_normals_kernel"
        elif parallel and not use_normals:
            kernel_name = "project_parallel_kernel"
        elif not parallel and use_normals:
            kernel_name = "project_conebeam_normals_kernel"
        else:
            kernel_name = "project_conebeam_kernel"

        print (f"Launching kernel: {kernel_name}")
        args = tuple(args)
        time, _ = cfg.BACKEND.pipeline.launchKernel(kernel_name, grid_size, block_size, args)

        return image.reshape(camera.shape).get()
    
    # def visualize_bvh(self, plotter, mapto="id", cmap="viridis"): # Changed default cmap
    #     """Visualize the bounding volume hierarchy."""
    #     if self._tree is None:
    #         raise ValueError("Tree is not built yet")

    #     # Ensure matplotlib and its components are imported
    #     # import matplotlib.colors as mcolors
    #     # import matplotlib.pyplot as plt # Already imported above for standalone

    #     cmapper = plt.get_cmap(cmap)

    #     # vertices = self._tree["vertices"] # Not used in this snippet directly for coloring
    #     bbMin = self._tree["bbMin"]
    #     bbMax = self._tree["bbMax"]
    #     nb_keys = len(self._triangles) // 3

    #     if nb_keys == 0:
    #         print("No keys to visualize.")
    #         return

    #     # Get the bounding boxes data
    #     bbMin_np = bbMin.get() # Assuming .get() returns a numpy array
    #     bbMax_np = bbMax.get() # Assuming .get() returns a numpy array

    #     if mapto == "id":
    #         minv, maxv = 0, nb_keys # Max value for normalization is nb_keys
    #         values = xp.arange(minv, maxv, dtype=float)
    #         if nb_keys == 1: # Special case for a single item to avoid division by zero in norm if maxv=minv
    #              minv, maxv = 0, 1 
    #     elif mapto == "depth":
    #         # --- ACCURATE DEPTH CALCULATION NEEDED ---
    #         # The following is a placeholder. You need to replace this with
    #         # actual depth data for each of the 'nb_keys' leaf nodes.
    #         # This data should come from your BVH tree structure.
    #         # Example: leaf_node_depths = self._tree.get_leaf_depths_array()
            
    #         # Placeholder / Example:
    #         if hasattr(self, '_leaf_depths_example') and len(self._leaf_depths_example) == nb_keys:
    #             leaf_node_depths = self._leaf_depths_example
    #         else:
    #             # Fallback if placeholder not set up or incorrect size - User must provide real data
    #             print(f"Warning: Using dummy depth data for 'mapto=depth'. Replace with actual depths for meaningful visualization.")
    #             # Create some plausible looking depth data if nb_keys > 0
    #             base_depth = xp.log2(nb_keys) if nb_keys > 1 else 1.0
    #             leaf_node_depths = xp.random.uniform(base_depth * 0.8, base_depth * 1.5, size=nb_keys)
    #             if nb_keys == 1: leaf_node_depths = xp.array([1.0])


    #         values = xp.array(leaf_node_depths, dtype=float)
    #         if len(values) == 0 : # Should not happen if nb_keys > 0
    #             minv, maxv = 0,1
    #         elif xp.all(values == values[0]): # All values are the same
    #             minv = values[0] - 0.5
    #             maxv = values[0] + 0.5
    #         else:
    #             minv, maxv = xp.min(values), xp.max(values)
    #         # --- END ACCURATE DEPTH CALCULATION SECTION ---

    #     elif mapto == "volume":
    #         lengths = bbMax_np[nb_keys : 2 * nb_keys, :3] - bbMin_np[nb_keys : 2 * nb_keys, :3]
    #         # Ensure we only use the leaf node bounding boxes for volume calculation
    #         # The loop iterates i from nb_keys to 2*nb_keys-1.
    #         # So bbMin_np[i,:] and bbMax_np[i,:] are used.
    #         # The 'lengths' should correspond to these.
    #         # Let's assume bbMin_np and bbMax_np are structured such that
    #         # indices nb_keys to 2*nb_keys-1 are the leaf nodes.
            
    #         if bbMin_np.shape[0] < 2 * nb_keys:
    #              raise ValueError("bbMin_np/bbMax_np arrays are not large enough for leaf node indexing.")

    #         # Calculate volumes for the relevant segment of bbMin_np/bbMax_np
    #         leaf_bbMin_np = bbMin_np[nb_keys : 2 * nb_keys]
    #         leaf_bbMax_np = bbMax_np[nb_keys : 2 * nb_keys]
            
    #         volumes_raw = xp.prod(leaf_bbMax_np[:, :3] - leaf_bbMin_np[:, :3], axis=1)
    #         values = xp.log(volumes_raw + 1e-9) # add small epsilon to avoid log(0)
            
    #         if len(values) == 0:
    #             minv,maxv = 0,1
    #         elif xp.all(values == values[0]): # All values are the same
    #             minv = values[0] - 0.5
    #             maxv = values[0] + 0.5
    #         else:
    #             minv, maxv = xp.min(values), xp.max(values)
    #     else:
    #         raise ValueError(f"Invalid mapto value: {mapto}")

    #     # Normalize the values
    #     if minv == maxv : # Avoid error if all values are identical
    #         norm = mcolors.Normalize(vmin=minv - 1e-6, vmax=maxv + 1e-6) # Create a tiny range
    #     else:
    #         norm = mcolors.Normalize(vmin=minv, vmax=maxv)
            
    #     sm = plt.cm.ScalarMappable(cmap=cmapper, norm=norm) # Use cmapper which is plt.get_cmap(cmap)

    #     # Visualize the leaf nodes
    #     for i in range(nb_keys, 2 * nb_keys):
    #         idx = i - nb_keys  # Index in the volumes/depths/ids array (0 to nb_keys-1)
            
    #         # Ensure idx is within bounds for `values`
    #         if idx >= len(values):
    #             print(f"Warning: idx {idx} is out of bounds for values array of_len {len(values)}. Skipping.")
    #             continue

    #         bounds = (
    #             bbMin_np[i, 0],  # xMin
    #             bbMax_np[i, 0],  # xMax
    #             bbMin_np[i, 1],  # yMin
    #             bbMax_np[i, 1],  # yMax
    #             bbMin_np[i, 2],  # zMin
    #             bbMax_np[i, 2],  # zMax
    #         )
            
    #         # Ensure bounds are valid
    #         if not (bbMax_np[i,0] >= bbMin_np[i,0] and \
    #                 bbMax_np[i,1] >= bbMin_np[i,1] and \
    #                 bbMax_np[i,2] >= bbMin_np[i,2]):
    #             # print(f"Warning: Invalid bounds for box {idx} at tree index {i}. Skipping.")
    #             # print(f"Bounds: {bounds}")
    #             # It might be better to create a tiny valid box or skip
    #             continue # Skip invalid boxes


    #         color_to_use = sm.to_rgba(values[idx])
    #         plotter.add_mesh(pv.Box(bounds=bounds), color=color_to_use, show_edges=True, opacity=0.9, name=f"Box {idx}", label=f"Box {idx}")

class LegacyCpuAccelerator(AcceleratorBase):
    """The fallback CPU-based projection strategy (based on OpenCL version)."""
    
    def build(self, **kwargs):
        """
        Prepares the mesh for projection by applying transformations and sorting.
        This version takes no arguments as it operates on the stored mesh.
        """
        self.mesh.transform()
        self.mesh.sort()

    def project(self, shape, pixel_size, /, *, t=None, offset=None, **kwargs):
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
        if out is None:
            out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_float)

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