"""Analytic regressions for CUDA intersection, thickness, and pixel sampling.

Run with SYRIS_REQUIRE_CUDA=1 to make a missing device an error instead of a skip.
The tests compile production sources and exercise both device functions and the
Python accelerator; no rendered image is used as a reference.
"""

import unittest

import numpy as np

from .cuda_projection_util import ProjectionHarness, box


class ProjectionCases:
    mixed_precision = False

    @classmethod
    def setUpClass(cls):
        cls.gpu = ProjectionHarness(
            cls.double,
            mixed_precision=cls.mixed_precision,
            hit_capacity=getattr(cls, "hit_capacity", None),
        )

    @property
    def rtol(self):
        return 3e-5 if not self.double else 2e-12

    def test_shared_edges_survive_winding_and_scale_changes(self):
        rng = np.random.default_rng(190753)
        b = rng.uniform(0.1, 1, (512, 2)).astype(self.gpu.dtype)
        b = np.column_stack([b, np.zeros(len(b))])
        a = 0.01 * np.column_stack([-b[:, 1], b[:, 0], np.zeros(len(b))])
        pairs = np.stack([np.stack([a, b, -b], axis=1), np.stack([-a, -b, b], axis=1)], axis=1)
        for scale in (1e-9, 1e-3, 1, 1e3):
            for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
                with self.subTest(scale=scale, order=order):
                    triangles = (pairs[:, :, order] * scale).reshape(-1, 3, 3)
                    origins = np.tile([0, 0, -scale], (len(triangles), 1))
                    directions = np.tile([0, 0, 1], (len(triangles), 1))
                    flags, distances = self.gpu.triangles(origins, directions, triangles)
                    # +/-b share the exact midpoint (0,0,0), also after quantization.
                    self.assertTrue(np.all(flags[:, 0].reshape(-1, 2).any(axis=1)))
                    self.assertTrue(np.all(flags[flags[:, 0] == 1, 1]))
                    np.testing.assert_allclose(distances[flags[:, 0] == 1], scale, rtol=self.rtol)

    def test_oblique_shared_edges_with_float32_vertices(self):
        rng = np.random.default_rng(1842)
        b = rng.uniform(0.1, 1, (2048, 2)).astype(np.float32)
        b = np.column_stack([b, np.zeros(len(b))])
        a = np.column_stack([-b[:, 1], b[:, 0], np.zeros(len(b))])
        triangles = np.stack(
            [np.stack([a, b, -b], axis=1), np.stack([-a, -b, b], axis=1)], axis=1
        ).reshape(-1, 3, 3)
        d = np.column_stack([rng.uniform(-0.4, 0.4, (len(b), 2)), np.ones(len(b))])
        d = (d / np.linalg.norm(d, axis=1)[:, None]).astype(self.gpu.dtype)
        d = np.repeat(d, 2, axis=0)
        flags, _ = self.gpu.triangles(-d, d, triangles)
        self.assertTrue(np.all(flags[:, 0].reshape(-1, 2).any(axis=1)))

    def test_clear_misses_and_degenerate_triangles_are_rejected(self):
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]] * 4)
        triangles[3] = [[0, 0, 0], [0.5, 0.5, 0], [1, 1, 0]]
        flags, distances = self.gpu.triangles(
            [[0.2, 0.3, -1], [0.5, 0.5001, -1], [0.2, 0.3, 1], [0.5, 0.5, -1]],
            [[0, 0, 1]] * 4,
            triangles,
        )
        np.testing.assert_array_equal(flags[:, 0], [1, 0, 0, 0])
        np.testing.assert_allclose(distances[0], 1, rtol=self.rtol)

    def test_box_clipping_preserves_precision_and_parallel_faces(self):
        z = 1 + (4e-8 if self.double else 4e-5)
        lower = [[-1, -1, z]] * 4 + [[-1, -1, -1]]
        upper = [[1, 1, z]] * 4 + [[1, 1, 1]]
        flags = self.gpu.boxes(
            [[0, 0, 0], [1, 1, 0], [2, 0, 0], [0, 0, 2], [0, 0, 0]],
            [[0, -0.0, 1], [-0.0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 1, 0]],
            lower,
            upper,
            [[z, z], [0, 4], [0, 4], [0, 4], [0, 4]],
        )
        np.testing.assert_array_equal(flags, [1, 1, 0, 1, 1])

    def test_box_shared_diagonals_and_interior_rays(self):
        accel = self.gpu.build(box())
        origins = [[0, 0, -1], [0.123, 0.123, -1], [0.1, 0.2, -1], [0.8, 0, -1]]
        result = self.gpu.trace(accel, origins, [[0, 0, 1]] * len(origins))
        np.testing.assert_allclose(result, [[1, 1], [1, 1], [1, 1], [0, 0]], rtol=self.rtol)

    def test_ray_starting_inside_material(self):
        result = self.gpu.trace(self.gpu.build(box()), [[0.1, 0.2, 0.25]], [[0, 0, 1]])
        np.testing.assert_allclose(result, 0.75, rtol=self.rtol)

    def test_oblique_rays_match_analytic_rotated_box(self):
        rng = np.random.default_rng(1771)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        translation = np.array([1.25, -0.75, 0.4])
        local_origins = rng.uniform(-2, 2, (512, 3))
        targets = rng.uniform(-0.4, 0.4, (512, 3))
        directions = targets - local_origins
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        # Analytic slab lengths in the box's local frame, independent of its
        # tessellation and of the CUDA triangle predicates.
        near = (-0.5 - local_origins) / directions
        far = (0.5 - local_origins) / directions
        entry = np.maximum(np.minimum(near, far).max(axis=1), 0)
        exit = np.maximum(near, far).min(axis=1)
        expected = np.maximum(exit - entry, 0)
        vertices = box(z0=-0.5, z1=0.5) @ rotation.T + translation
        actual = self.gpu.trace(
            self.gpu.build(vertices),
            local_origins @ rotation.T + translation,
            directions @ rotation.T,
        )
        np.testing.assert_allclose(
            actual,
            np.repeat(expected[:, None], 2, axis=1),
            rtol=self.rtol,
            atol=2e-6 if not self.double else 1e-13,
        )

    def test_thin_layer_is_independent_of_external_ray_origin(self):
        accel = self.gpu.build(box(z1=5e-6))
        origins = [[0.1, 0.2, -distance] for distance in (0.01, 1, 100)]
        result = self.gpu.trace(accel, origins, [[0, 0, 1]] * len(origins))
        np.testing.assert_allclose(result, 5e-6, rtol=2e-4 if not self.double else 1e-9)

    def test_candidate_and_actual_hit_counts_do_not_truncate_layers(self):
        for layers in (80, 140):
            with self.subTest(layers=layers):
                vertices = np.concatenate(
                    [box(z0=i * 0.02, z1=i * 0.02 + 0.01) for i in range(layers)]
                )
                accel = self.gpu.build(vertices)
                result = self.gpu.trace(accel, [[0.1, 0.2, -1]], [[0, 0, 1]])
                np.testing.assert_allclose(result, layers * 0.01, rtol=self.rtol)

    def test_common_large_and_unbounded_hit_paths_agree_on_shared_edges(self):
        # Four triangle hits per layer on the shared face diagonals. Exercise
        # both sides of the 32- and 256-record limits, plus interior rays.
        for layers in (8, 9, 16, 17, 64, 65, 140):
            with self.subTest(layers=layers):
                vertices = np.concatenate(
                    [box(z0=i * 0.02, z1=i * 0.02 + 0.01) for i in range(layers)]
                )
                accel = self.gpu.build(vertices)
                result = self.gpu.trace(accel, [[0, 0, -1], [0.1, 0.2, -1]], [[0, 0, 1]] * 2)
                np.testing.assert_allclose(result, layers * 0.01, rtol=self.rtol)

    def test_event_batches_keep_complete_duplicate_groups(self):
        # More than 256 hits but only two distinct distances. Duplicates must
        # neither consume the event batch nor toggle parity repeatedly.
        vertices = np.tile(box(), (150, 1, 1))
        accel = self.gpu.build(vertices)
        result = self.gpu.trace(
            accel, [[0, 0, -1], [0.1, 0.2, -1], [0.1, 0.2, 0.25]], [[0, 0, 1]] * 3
        )
        np.testing.assert_allclose(result, [[1, 1], [1, 1], [0.75, 0.75]], rtol=self.rtol)

    def test_event_batches_order_reversed_rays_and_shuffled_thin_layers(self):
        rng = np.random.default_rng(731)
        widths = rng.uniform(1e-5, 0.01, 160)
        vertices = np.concatenate(
            [box(z0=i * 0.02, z1=i * 0.02 + width) for i, width in enumerate(widths)]
        )
        vertices = vertices[rng.permutation(len(vertices))]
        accel = self.gpu.build(vertices)
        result = self.gpu.trace(
            accel,
            [[0.1, 0.2, -1], [0.1, 0.2, 4], [0.1, 0.2, widths[0] / 2]],
            [[0, 0, 1], [0, 0, -1], [0, 0, 1]],
        )
        expected = np.array([widths.sum(), widths.sum(), widths.sum() - widths[0] / 2])
        np.testing.assert_allclose(result, np.repeat(expected[:, None], 2, axis=1), rtol=self.rtol)

    def test_staged_sampling_preserves_local_rays_and_refinement(self):
        vertices = np.concatenate(
            [box(-0.7, 0.6, -0.8, 0.5, 0, 1.3), box(-0.2, 0.3, -0.3, 0.2, 0.4, 0.8)[:, ::-1]]
        )
        rotation, _ = np.linalg.qr(np.random.default_rng(192).normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        for normals in (False, True):
            accel = self.gpu.build(vertices, use_normals=normals)
            for parallel in (False, True):
                for basis in (np.eye(3), rotation):
                    for shape in ((7, 11), (1, 9), (9, 1)):
                        for depth, tolerance in ((1, 1e6), (1, 0.02), (2 if self.double else 3, 0)):
                            with self.subTest(
                                normals=normals,
                                parallel=parallel,
                                shape=shape,
                                depth=depth,
                                tolerance=tolerance,
                            ):
                                kwargs = dict(
                                    parallel=parallel,
                                    depth=depth,
                                    shape=shape,
                                    basis=basis,
                                    origin=(-0.9231, -0.8763, -1.3127),
                                    pixel_size=(0.1731, 0.2117),
                                    source=(0, 0, -3),
                                    abs_tolerance=0,
                                    rel_tolerance=tolerance,
                                )
                                local = self.gpu.project_image(accel, **kwargs)
                                staged = self.gpu.project_image(
                                    accel, sampling_mode="staged", **kwargs
                                )
                                np.testing.assert_array_equal(staged, local)

    def test_staged_sampling_preserves_nonfinite_results(self):
        # Outward orientation reversed: the invalid crossing sequence must not
        # be accepted as a finite path length by the coarse classification.
        accel = self.gpu.build(box()[:, ::-1])
        for parallel in (False, True):
            for tolerance in (0, 1e6):
                kwargs = dict(
                    depth=1,
                    parallel=parallel,
                    shape=(3, 5),
                    origin=(-0.4, -0.3, -1),
                    pixel_size=(0.15, 0.2),
                    source=(0, 0, -3),
                    rel_tolerance=tolerance,
                )
                local = self.gpu.project_image(accel, **kwargs)
                self.assertTrue(np.isnan(local).any())
                staged = self.gpu.project_image(accel, sampling_mode="staged", **kwargs)
                np.testing.assert_array_equal(staged, local)

    def test_closed_cavity_preserves_empty_interval(self):
        vertices = np.concatenate([box(z1=3), box(-0.4, 0.4, -0.4, 0.4, 1, 2)[:, ::-1]])
        result = self.gpu.trace(self.gpu.build(vertices), [[0.1, 0.2, -1]], [[0, 0, 1]])
        np.testing.assert_allclose(result, 2, rtol=self.rtol)

    def test_conebeam_center_ray_matches_parallel_thickness(self):
        for normals in (False, True):
            with self.subTest(normals=normals):
                accel = self.gpu.build(box(), use_normals=normals)
                result = self.gpu.project(
                    accel,
                    parallel=False,
                    origin=(-0.1, -0.1, 3),
                    pixel_size=(0.2, 0.2),
                    source=(0, 0, -3),
                )
                np.testing.assert_allclose(result, 1, rtol=self.rtol)

    def test_oblique_conebeam_measures_path_length(self):
        # Detector center (2.1, 1.1, 3), source (0, 0, -3): L = dz / cos(theta).
        expected = np.linalg.norm([2.1, 1.1, 6]) / 6
        for normals in (False, True):
            with self.subTest(normals=normals):
                accel = self.gpu.build(box(-3, 3, -3, 3), use_normals=normals)
                result = self.gpu.project(
                    accel,
                    parallel=False,
                    origin=(2, 1, 3),
                    pixel_size=(0.2, 0.2),
                    source=(0, 0, -3),
                )
                np.testing.assert_allclose(result, expected, rtol=self.rtol)

    def test_adaptive_conebeam_integrates_oblique_slab(self):
        nodes, weights = np.polynomial.legendre.leggauss(12)
        x = 2.1 + 0.1 * nodes[:, None]
        y = 1.1 + 0.1 * nodes[None, :]
        reference = (
            np.sum(np.sqrt(1 + (x * x + y * y) / 36) * weights[:, None] * weights[None, :]) / 4
        )
        for normals in (False, True):
            with self.subTest(normals=normals):
                accel = self.gpu.build(box(-3, 3, -3, 3), use_normals=normals)
                result = self.gpu.project(
                    accel,
                    parallel=False,
                    depth=2 if self.double else 3,
                    origin=(2, 1, 3),
                    pixel_size=(0.2, 0.2),
                    source=(0, 0, -3),
                    abs_tolerance=0,
                    rel_tolerance=0,
                )
                np.testing.assert_allclose(result, reference, rtol=1e-5)

    def test_adaptive_linear_thickness_at_every_supported_depth(self):
        vertices = box(-1, 2, -1, 2)
        vertices[:, :, 2] = np.where(vertices[:, :, 2] > 0, 2 + vertices[:, :, 1], 0)
        for normals in (False, True):
            accel = self.gpu.build(vertices, use_normals=normals)
            for depth in range(1, (2 if self.double else 3) + 1):
                with self.subTest(normals=normals, depth=depth):
                    result = self.gpu.project(accel, depth=depth, abs_tolerance=0, rel_tolerance=0)
                    np.testing.assert_allclose(result, 2.5, rtol=self.rtol)

    def test_python_tolerances_control_refinement(self):
        accel = self.gpu.build(box(-0.1, 0.3, -0.1, 1.1))
        coarse = self.gpu.project(accel, depth=2, rel_tolerance=1e6)
        refined = self.gpu.project(accel, depth=2, abs_tolerance=1e-5, rel_tolerance=0.02)
        self.assertEqual(coarse, 0)
        self.assertGreater(refined, 0.2)
        self.assertLess(refined, 0.4)

    def test_normals_follow_mesh_transform(self):
        import quantities as q

        from syris.bodies.accelerators import BvhCupyAccelerator
        from syris.bodies.mesh import Mesh
        from syris.geometry import Trajectory

        vertices = box().astype(self.gpu.dtype)
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        for supplied in ("computed", "array", "quantity"):
            with self.subTest(supplied=supplied), self.gpu.configured():
                supplied_normals = {"computed": None, "array": normals, "quantity": normals * q.um}[
                    supplied
                ]
                mesh = Mesh(
                    vertices.reshape(-1, 3).T * q.um,
                    Trajectory([(0, 0, 0)] * q.um),
                    center=None,
                    use_normals=True,
                    normals=supplied_normals,
                )
                mesh.transform()
                original_normals = np.asarray(mesh.normals).copy()
                mesh.transform_matrix = np.diag([1, -1, -1, 1]).astype(self.gpu.dtype)
                mesh.transform()
                np.testing.assert_allclose(
                    np.asarray(mesh.normals), original_normals @ np.diag([1, -1, -1])
                )
                accel = BvhCupyAccelerator(mesh)
                accel.build()
                np.testing.assert_allclose(
                    self.gpu.trace(accel, [[0.1, 0.2, -2]], [[0, 0, 1]]), 1, rtol=self.rtol
                )

    def test_morton_leaves_map_back_to_original_triangles(self):
        vertices = np.random.default_rng(741).uniform(-2, 2, (512, 3, 3)).astype(self.gpu.dtype)
        tree = self.gpu.build(vertices).tree
        indices = tree["indices"].get()
        encoded = tree["vertices"].get()[:, :3].reshape(-1, 3, 3)
        self.assertEqual(indices.dtype, np.uint32)
        np.testing.assert_array_equal(
            tree["bbMin"].get()[: len(vertices), :3], encoded[indices].min(axis=1)
        )
        np.testing.assert_array_equal(
            tree["bbMax"].get()[: len(vertices), :3], encoded[indices].max(axis=1)
        )

    def test_single_triangle_and_planar_bounds_build(self):
        vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        tree = self.gpu.build(vertices).tree
        self.assertEqual(int(tree["rope"].get()[0]), -1)
        np.testing.assert_array_equal(tree["bbMin"].get()[0, :3], [-0.5, -0.5, 0])
        np.testing.assert_array_equal(tree["bbMax"].get()[0, :3], [0.5, 0.5, 0])

    def test_invalid_sampling_settings_are_rejected(self):
        accel = self.gpu.build(box())
        for depth in (-1, 4, 1.5):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.gpu.project(accel, depth=depth)
        for kwargs in ({"abs_tolerance": -1}, {"rel_tolerance": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.gpu.project(accel, **kwargs)
        with self.assertRaises(ValueError):
            self.gpu.project(accel, sampling_mode="unknown")

    def test_bvh_published_bounds_contain_both_children(self):
        rng = np.random.default_rng(491)
        vertices = rng.uniform(-2, 2, (1024, 3, 3))
        for repeat in range(3):
            with self.subTest(repeat=repeat):
                tree = self.gpu.build(vertices).tree
                lo, hi = tree["bbMin"].get()[:, :3], tree["bbMax"].get()[:, :3]
                left, rope = tree["left"].get(), tree["rope"].get()
                n = len(vertices)
                seen = set()
                stack = [n]
                while stack:
                    node = stack.pop()
                    self.assertNotIn(node, seen)
                    seen.add(node)
                    if node < n:
                        continue
                    a = int(left[node])
                    b = int(rope[a])
                    self.assertTrue(0 <= a < 2 * n and 0 <= b < 2 * n)
                    self.assertTrue(np.all(lo[node] <= np.minimum(lo[a], lo[b])))
                    self.assertTrue(np.all(hi[node] >= np.maximum(hi[a], hi[b])))
                    stack.extend([a, b])
                self.assertEqual(len(seen), 2 * n - 1)


class TestCudaProjectionFloat32(ProjectionCases, unittest.TestCase):
    double = False


class TestCudaProjectionFloat64(ProjectionCases, unittest.TestCase):
    double = True


class TestCudaProjectionSmallHitBuffer(ProjectionCases, unittest.TestCase):
    double = False
    hit_capacity = 32


class TestCudaProjectionMixed(ProjectionCases, unittest.TestCase):
    double = False
    mixed_precision = True

    def test_distinct_crossings_survive_float32_hit_distance_collision(self):
        # A separate component puts the ray's rebased origin near z=-1.
        # Both hits on the thin slab then round to the SAME Float32 distance,
        # even though every input vertex and the final thickness fit Float32.
        thickness = 2**-28
        vertices = np.concatenate(
            [
                box(x0=-4, x1=-3, z0=-1, z1=1),
                box(z0=0, z1=thickness),
            ]
        )
        result = self.gpu.trace(self.gpu.build(vertices), [[0.1, 0.2, -2]], [[0, 0, 1]])
        np.testing.assert_allclose(result, thickness, rtol=1e-6, atol=0)

    def test_triangle_coordinates_are_promoted_before_subtracting_ray_origin(self):
        # All inputs are exactly representable in Float32. Subtracting the
        # distant origin in Float32 first collapses the projected triangle;
        # the ray itself passes through its interior at t = distance.
        size, distance = 2**-8, 2**20
        triangle = np.array([[-size, -size, 0], [size, -size, 0], [0, size, 0]])
        triangles = np.array([triangle, triangle[::-1], triangle])
        flags, distances = self.gpu.triangles(
            [[distance, 0, -distance], [distance, 0, -distance], [distance, 4 * size, -distance]],
            [[-1, 0, 1]] * 3,
            triangles,
        )
        np.testing.assert_array_equal(flags[:, 0], [1, 1, 0])
        np.testing.assert_array_equal(distances[:2], distance)
        self.assertEqual(distances.dtype, np.float32)


class TestMixedPrecisionConfiguration(unittest.TestCase):
    def test_incompatible_modes_are_rejected(self):
        import syris

        for kwargs in (
            {"compute_backend": "opencl"},
            {"compute_backend": "cuda", "double_precision": True},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "mixed_precision requires"),
            ):
                syris.init(mixed_precision=True, **kwargs)


class TestCudaModuleCache(unittest.TestCase):
    def test_changed_header_recompiles_the_module(self):
        import tempfile
        from pathlib import Path

        from .cuda_projection_util import cuda_or_skip

        cp = cuda_or_skip()
        from syris.gpu.cuda_utils import CudaPipeline

        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "value.cuh"
            source = Path(directory) / "value.cu"
            source.write_text(
                '#include "value.cuh"\nextern "C" __global__ void value(int *out) { *out = VALUE; }'
            )
            for expected in (17, 29):
                header.write_text(f"#define VALUE {expected}\n")
                pipeline = CudaPipeline([directory])
                pipeline.readModuleFromFiles("cache_test", [source])
                kernel = pipeline.getKernelFromModule("cache_test", "value")
                out = cp.empty(1, dtype=np.int32)
                kernel((1,), (1,), (out,))
                self.assertEqual(int(out.get()[0]), expected)
