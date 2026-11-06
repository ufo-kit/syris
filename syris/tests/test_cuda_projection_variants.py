"""Image-level checks for research pipelines outside ordinary initialization."""

import unittest

import numpy as np

from .cuda_projection_util import ProjectionHarness, box


class ProjectionVariantCases:
    double = False
    mixed = False

    @classmethod
    def setUpClass(cls):
        cls.gpu = ProjectionHarness(cls.double, mixed_precision=cls.mixed)
        from benchmarks.cuda_projection import DeferredSignedPipeline, DepthSpecializedPipeline

        options = ["-D__FP_T_D__"] if cls.double else ["-DSYRIS_MIXED_PRECISION"] if cls.mixed else []
        cls.deferred = DeferredSignedPipeline(options)
        cls.specialized = DepthSpecializedPipeline(options, [0, 1])

    def compare(self, accel, pipeline, **kwargs):
        accel.pipeline = self.gpu.pipeline
        expected = self.gpu.project_image(accel, **kwargs)
        accel.pipeline = pipeline
        actual = self.gpu.project_image(accel, **kwargs)
        # Different compensated summation order can change a Float64 final bit.
        # The finite/zero masks and sampling settings must still be preserved.
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_array_equal(actual == 0, expected == 0)
        np.testing.assert_allclose(actual, expected, rtol=8 * np.finfo(actual.dtype).eps,
                                   atol=0, equal_nan=True)
        self.assertFalse(np.isinf(actual).any(), "Internal retry markers must be consumed")
        return expected, actual

    def test_depth_specialization_keeps_samples_and_exact_output(self):
        vertices = box(-0.4, 0.6, -0.4, 0.6)
        vertices[:, :, 2] = np.where(vertices[:, :, 2] > 0, 2 + vertices[:, :, 1], 0)
        for normals in (False, True):
            accel = self.gpu.build(vertices, use_normals=normals)
            for parallel in (False, True):
                for shape in ((1, 5), (5, 1), (3, 5)):
                    for depth in (0, 1):
                        for tolerance in (0, 1e6):
                            with self.subTest(normals=normals, parallel=parallel, shape=shape,
                                              depth=depth, tolerance=tolerance):
                                expected, actual = self.compare(
                                    accel, self.specialized, parallel=parallel, shape=shape,
                                    depth=depth, origin=(-0.5, -0.4, -1),
                                    pixel_size=(0.1731, 0.2117), source=(0, 0, -3),
                                    abs_tolerance=0, rel_tolerance=tolerance)
                                np.testing.assert_array_equal(actual, expected)

    def test_depth_specialization_rejects_uncompiled_depth(self):
        accel = self.gpu.build(box())
        accel.pipeline = self.specialized
        with self.assertRaisesRegex(ValueError, "No kernel compiled"):
            self.gpu.project_image(accel, depth=2)

    def test_deferred_cavity_matches_at_every_supported_depth(self):
        vertices = np.concatenate([box(z1=3),
                                   box(-0.4, 0.4, -0.4, 0.4, 1, 2)[:, ::-1]])
        accel = self.gpu.build(vertices)
        for parallel in (False, True):
            for depth in range((2 if self.double else 3) + 1):
                with self.subTest(parallel=parallel, depth=depth):
                    self.compare(accel, self.deferred, depth=depth, parallel=parallel,
                                 shape=(3, 5), origin=(-0.5, -0.4, -1),
                                 pixel_size=(0.1731, 0.2117), source=(0, 0, -3),
                                 abs_tolerance=0, rel_tolerance=0)

    def test_deferred_empty_partial_and_full_retry_queues(self):
        accel = self.gpu.build(box())
        # First case has no diagonal/edge hits, the second crosses the face
        # triangulation diagonal, and the last is entirely a diagonal hit.
        cases = [((1, 1), (-0.1, 0.0, -1), (0.1, 0.1), 0),
                 ((3, 3), (-0.375, -0.375, -1), (0.25, 0.25), 3),
                 ((1, 1), (-0.1, -0.1, -1), (0.2, 0.2), 1)]
        for shape, origin, pixel_size, retries in cases:
            with self.subTest(retries=retries):
                self.compare(accel, self.deferred, shape=shape, origin=origin,
                             pixel_size=pixel_size)
                self.assertEqual(int(self.deferred.last_retry_count.get()[0]), retries)

    def test_deferred_invalid_crossings_remain_nan(self):
        accel = self.gpu.build(box()[:, ::-1])
        for parallel in (False, True):
            for depth in (0, 1):
                for tolerance in (0, 1e6):
                    with self.subTest(parallel=parallel, depth=depth, tolerance=tolerance):
                        expected, actual = self.compare(
                            accel, self.deferred, parallel=parallel, depth=depth,
                            shape=(3, 5), origin=(-0.4, -0.3, -1),
                            pixel_size=(0.15, 0.2), source=(0, 0, -3),
                            abs_tolerance=0, rel_tolerance=tolerance)
                        self.assertTrue(np.isnan(actual).any())
                        np.testing.assert_array_equal(actual, expected)


class TestProjectionVariantsFloat32(ProjectionVariantCases, unittest.TestCase):
    pass


class TestProjectionVariantsFloat64(ProjectionVariantCases, unittest.TestCase):
    double = True


class TestProjectionVariantsMixed(ProjectionVariantCases, unittest.TestCase):
    mixed = True
