"""Conditional signed-length prototype, including explicit contract limits."""

import unittest

import numpy as np

from .cuda_projection_util import ProjectionHarness, box
from .test_cuda_projection import ProjectionCases


class SignedProjectionCases(ProjectionCases):
    @classmethod
    def setUpClass(cls):
        cls.gpu = ProjectionHarness(
            cls.double, mixed_precision=cls.mixed_precision, signed_thickness=True
        )

    def test_event_batches_keep_complete_duplicate_groups(self):
        # Duplicated shells violate the signed contract, but exact boundary
        # rays must still take the sorted fallback and deduplicate correctly.
        accel = self.gpu.build(np.tile(box(), (150, 1, 1)))
        actual = self.gpu.trace(accel, [[0, 0, -1], [0, 0, 0.25]], [[0, 0, 1]] * 2)
        np.testing.assert_allclose(actual, [[1, 1], [0.75, 0.75]], rtol=self.rtol)

    def test_signed_contract_excludes_overlapping_shells(self):
        # This counterexample is why the mode must be explicitly opted into:
        # two unit-thickness overlapping boxes have union thickness 1.5 but
        # signed winding-weighted thickness 2. Balanced endpoints cannot detect
        # the problem. This must never be advertised as a general mesh mode.
        accel = self.gpu.build(np.concatenate([box(), box(z0=0.5, z1=1.5)]))
        actual = self.gpu.trace(accel, [[0.1, 0.2, -1]], [[0, 0, 1]])
        self.assertAlmostEqual(actual[0, 1], 2, delta=2 * self.rtol)
        self.assertGreater(abs(actual[0, 1] - 1.5), 0.4)

    def test_nested_material_island_inside_cavity_and_internal_origins(self):
        vertices = np.concatenate(
            [box(z1=5), box(-0.4, 0.4, -0.4, 0.4, 1, 4)[:, ::-1], box(-0.2, 0.2, -0.2, 0.2, 2, 3)]
        )
        origins = [
            [0.1, 0.17, -1],
            [0.3, 0.17, -1],
            [0.45, 0.17, -1],
            [0.1, 0.17, 0.5],
            [0.1, 0.17, 1.5],
            [0.1, 0.17, 2.5],
            [0.1, 0.17, 6],
        ]
        directions = [[0, 0, 1]] * 6 + [[0, 0, -1]]
        expected = np.array([3, 2, 5, 2.5, 2, 1.5, 3])
        actual = self.gpu.trace(self.gpu.build(vertices), origins, directions)
        np.testing.assert_allclose(actual, np.repeat(expected[:, None], 2, axis=1), rtol=self.rtol)

    def test_many_disjoint_thin_layers_match_ordered_reference(self):
        rng = np.random.default_rng(20261106)
        for layers in (20, 80, 140, 600):
            with self.subTest(layers=layers):
                widths = rng.uniform(1e-5, 0.007, layers)
                vertices = np.concatenate(
                    [box(z0=i * 0.01, z1=i * 0.01 + w) for i, w in enumerate(widths)]
                )
                vertices = vertices[rng.permutation(len(vertices))]
                result = self.gpu.trace(
                    self.gpu.build(vertices),
                    [[0.1, 0.2, -1], [0.1, 0.2, layers * 0.01 + 1]],
                    [[0, 0, 1], [0, 0, -1]],
                )
                np.testing.assert_allclose(result, widths.sum(), rtol=self.rtol)
                np.testing.assert_allclose(
                    result[:, 1], result[:, 0], rtol=0, atol=2e-14 if self.double else 0
                )


class TestSignedFloat32(SignedProjectionCases, unittest.TestCase):
    double = False


class TestSignedFloat64(SignedProjectionCases, unittest.TestCase):
    double = True


class TestSignedMixed(SignedProjectionCases, unittest.TestCase):
    double = False
    mixed_precision = True
