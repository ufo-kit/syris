"""Exact regressions for box scalar selection and spatial pixel scheduling."""

import unittest
from pathlib import Path

import numpy as np

from .cuda_projection_util import ProjectionHarness, box, cuda_or_skip


class ScalarSelectionCases:
    double = False
    mixed = False

    @classmethod
    def setUpClass(cls):
        cls.scalar = ProjectionHarness(
            cls.double, mixed_precision=cls.mixed, box_scalar_selection=2
        )
        cls.indexed = ProjectionHarness(
            cls.double, mixed_precision=cls.mixed, box_scalar_selection=False
        )

    def test_box_decisions_match_indexed_predicate(self):
        rng = np.random.default_rng(693571)
        n = 32768
        dtype = self.scalar.dtype
        lower = rng.uniform(-2, 0, (n, 3)).astype(dtype)
        upper = (lower + rng.uniform(0, 3, (n, 3))).astype(dtype)
        origins = rng.uniform(-3, 3, (n, 3)).astype(dtype)
        directions = rng.normal(size=(n, 3)).astype(dtype)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        # All dominant axes and signs, including both signs of parallel zero.
        for axis in range(3):
            for sign in (-1, 1):
                rows = np.arange(6 * axis + (0 if sign < 0 else 3), n, 18)
                directions[rows] = dtype(-0.0 if sign < 0 else 0.0)
                directions[rows, axis] = sign
        directions[2::19] = [1e-25, -1e-25, 1]
        origins[::7] = lower[::7]
        origins[1::7] = upper[1::7]
        origins[2::7] = np.nextafter(lower[2::7], dtype(-np.inf))
        origins[3::7] = np.nextafter(upper[3::7], dtype(np.inf))
        upper[::11, 2] = lower[::11, 2]  # Zero-width boxes.
        ranges = np.column_stack([np.zeros(n), rng.uniform(0, 5, n)]).astype(dtype)
        ranges[::13, 1] = 0
        for scale in (1e-30, 1e-9, 1, 1e9, 1e30):
            with self.subTest(scale=scale):
                args = (
                    origins * dtype(scale),
                    directions,
                    lower * dtype(scale),
                    upper * dtype(scale),
                    ranges * dtype(scale),
                )
                expected = self.indexed.boxes(*args)
                actual = self.scalar.boxes(*args)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.any())
                self.assertFalse(actual.all())


class TestScalarSelectionFloat32(ScalarSelectionCases, unittest.TestCase):
    pass


class TestScalarSelectionFloat64(ScalarSelectionCases, unittest.TestCase):
    double = True


class TestScalarSelectionMixed(ScalarSelectionCases, unittest.TestCase):
    mixed = True


class TestPixelSchedulerCoverage(unittest.TestCase):
    def test_every_pixel_is_claimed_once_including_partial_warps_and_tiles(self):
        cp = cuda_or_skip()
        source = Path(__file__).resolve().parents[1] / "gpu" / "cuda"
        code = r"""
        #include "PixelScheduler.cuh"
        extern "C" __global__ void visit(unsigned *counter, unsigned cols, unsigned rows,
                                        unsigned *visits) {
            uint2 size = make_uint2(cols, rows);
            unsigned index;
            while (nextProjectionPixel(counter, size, index)) {
                if (index >= size.x * size.y) continue;
                atomicAdd(visits + index, 1);
            }
        }
        """
        for width in (0, 1, 2, 4, 8, 16, 32):
            module = cp.RawModule(
                code=code,
                backend="nvcc",
                options=(f"-I{source}", f"-DSYRIS_WARP_TILE_WIDTH={width}", "--std=c++17"),
            )
            kernel = module.get_function("visit")
            for rows, cols in ((0, 0), (1, 1), (1, 37), (37, 1), (11, 19), (65, 67)):
                for block in ((1, 1, 1), (17, 1, 1), (33, 1, 1), (128, 1, 1), (16, 2, 1)):
                    with self.subTest(width=width, shape=(rows, cols), block=block):
                        visits = cp.zeros(rows * cols + 2, cp.uint32)
                        visits[0] = visits[-1] = 0x12345678
                        counter = cp.zeros(1, cp.uint32)
                        kernel(
                            (4, 1, 1),
                            block,
                            (counter, np.uint32(cols), np.uint32(rows), visits[1:-1]),
                        )
                        result = visits.get()
                        np.testing.assert_array_equal(result[1:-1], 1)
                        self.assertEqual(int(result[0]), 0x12345678)
                        self.assertEqual(int(result[-1]), 0x12345678)


class WarpTileCases:
    double = False
    mixed = False

    @classmethod
    def setUpClass(cls):
        cls.gpu = ProjectionHarness(
            cls.double, mixed_precision=cls.mixed, box_scalar_selection=2, warp_tile_width=0
        )
        from benchmarks.cuda_projection import experimental_pipeline

        options = (
            ["-D__FP_T_D__"] if cls.double else ["-DSYRIS_MIXED_PRECISION"] if cls.mixed else []
        )
        cls.tiles = {
            width: experimental_pipeline(options, f"tile{width}_axes") for width in (4, 8, 16, 32)
        }

    def compare(self, accel, **kwargs):
        accel.pipeline = self.gpu.pipeline
        expected = self.gpu.project_image(accel, **kwargs)
        for width, pipeline in self.tiles.items():
            with self.subTest(width=width):
                accel.pipeline = pipeline
                actual = self.gpu.project_image(accel, **kwargs)
                np.testing.assert_array_equal(actual, expected)
        return expected

    def test_partial_tiles_match_reference(self):
        vertices = box(-0.4, 0.6, -0.4, 0.6)
        vertices[:, :, 2] = np.where(vertices[:, :, 2] > 0, 2 + vertices[:, :, 1], 0)
        for normals in (False, True):
            accel = self.gpu.build(vertices, use_normals=normals)
            for parallel in (False, True):
                for shape in ((1, 1), (1, 37), (37, 1), (11, 19)):
                    for depth in (0, 1):
                        with self.subTest(
                            normals=normals, parallel=parallel, shape=shape, depth=depth
                        ):
                            self.compare(
                                accel,
                                parallel=parallel,
                                shape=shape,
                                depth=depth,
                                origin=(-0.5, -0.4, -1),
                                pixel_size=(0.1731, 0.2117),
                                source=(0, 0, -3),
                                abs_tolerance=0,
                                rel_tolerance=0,
                            )

    def test_deep_adaptive_cavity_keeps_sample_results(self):
        vertices = np.concatenate([box(z1=3), box(-0.4, 0.4, -0.4, 0.4, 1, 2)[:, ::-1]])
        accel = self.gpu.build(vertices)
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self.compare(
                    accel,
                    parallel=parallel,
                    depth=2 if self.double else 3,
                    shape=(7, 9),
                    origin=(-0.5, -0.4, -1),
                    pixel_size=(0.1731, 0.2117),
                    source=(0, 0, -3),
                    abs_tolerance=0,
                    rel_tolerance=0,
                )

    def test_overflow_and_invalid_pixels_keep_their_values(self):
        # Wide layers keep every cone ray inside the lateral bounds through all
        # 140 layers. The narrow case also checks side exits, including any NaNs
        # already produced by the reference; it need not overflow for every ray.
        layers = np.concatenate([box(-2, 2, -2, 2, i * 0.02, i * 0.02 + 0.01) for i in range(140)])
        narrow = np.concatenate([box(z0=i * 0.02, z1=i * 0.02 + 0.01) for i in range(140)])
        for name, vertices in (
            ("overflow", layers),
            ("side_exit", narrow),
            ("invalid", box()[:, ::-1]),
        ):
            accel = self.gpu.build(vertices)
            for parallel in (False, True):
                for depth in (0, 1):
                    with self.subTest(scene=name, parallel=parallel, depth=depth):
                        image = self.compare(
                            accel,
                            parallel=parallel,
                            depth=depth,
                            shape=(3, 5),
                            origin=(-0.4, -0.3, -1),
                            pixel_size=(0.15, 0.2),
                            source=(0, 0, -3),
                        )
                        if name == "overflow":
                            self.assertTrue(np.isfinite(image).all())
                            if depth == 0:
                                expected = np.full((3, 5), 1.4)
                                if not parallel:
                                    row, col = np.indices((3, 5))
                                    x = -0.4 + (col + 0.5) * 0.15
                                    y = -0.3 + (row + 0.5) * 0.2
                                    expected *= np.sqrt(x * x + y * y + 4) / 2
                                np.testing.assert_allclose(
                                    image, expected, rtol=128 * np.finfo(image.dtype).eps, atol=0
                                )
                        elif name == "invalid":
                            self.assertTrue(np.isnan(image).any())


class TestWarpTilesFloat32(WarpTileCases, unittest.TestCase):
    pass


class TestWarpTilesFloat64(WarpTileCases, unittest.TestCase):
    double = True


class TestWarpTilesMixed(WarpTileCases, unittest.TestCase):
    mixed = True


class TestWarpTileConfiguration(unittest.TestCase):
    def test_invalid_widths_and_backends_are_rejected(self):
        import syris

        for width in (-1, 3, 64, 1.5, "8", True):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "warp_tile_width"):
                syris.init(compute_backend="cuda", warp_tile_width=width)
        with self.assertRaisesRegex(ValueError, "warp_tile_width requires"):
            syris.init(compute_backend="opencl", warp_tile_width=8)

    def test_public_init_uses_tiles_and_preserves_projection(self):
        import logging

        import syris
        from syris import config as cfg

        cuda_or_skip()
        syris.init(compute_backend="cuda", warp_tile_width=4, loglevel=logging.ERROR)
        production = cfg.BACKEND.pipeline
        self.assertIn("-DSYRIS_WARP_TILE_WIDTH=4", production.opts)
        gpu = ProjectionHarness(box_scalar_selection=2, warp_tile_width=0)
        accel = gpu.build(box())
        kwargs = dict(
            shape=(11, 19),
            depth=1,
            origin=(-0.6, -0.4, -1),
            pixel_size=(0.08, 0.11),
            abs_tolerance=0,
            rel_tolerance=0,
        )
        expected = gpu.project_image(accel, **kwargs)
        accel.pipeline = production
        actual = gpu.project_image(accel, **kwargs)
        np.testing.assert_array_equal(actual, expected)
