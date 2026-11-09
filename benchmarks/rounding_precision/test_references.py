"""Analytic checks for the independent plane-intersection references."""

import unittest

import numpy as np

from syris.tests.cuda_projection_util import box

from .probe import decimal_reference
from .validate import plane_references


def soup(triangles):
    points = np.asarray(triangles).reshape(-1, 3)
    return points, np.arange(len(points)).reshape(-1, 3)


class TestReferences(unittest.TestCase):
    def test_units_and_disjoint_material_intervals(self):
        points, faces = soup(np.concatenate([box(), box(z0=2, z1=2.125)]))
        xy = [(0.12, -0.23), (0.2, 0.33), (2, 0)]
        expected_m = np.array([1.125e-6, 1.125e-6, 0])
        np.testing.assert_allclose(plane_references(points, faces, xy), expected_m, atol=1e-20)
        decimal_m = [decimal_reference(points, faces, pos) * 1e-6 for pos in xy]
        np.testing.assert_allclose(decimal_m, expected_m, atol=1e-20)

    def test_oblique_slab_path_length(self):
        points, faces = soup(box(x0=-2, x1=2, y0=-2, y1=2, z0=-0.5, z1=0.5))
        theta = np.pi / 6
        rotation = np.array(
            [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
        )
        points = points @ rotation.T
        xy = [(0.031, -0.12)]
        expected_m = 1e-6 / np.cos(theta)
        self.assertAlmostEqual(plane_references(points, faces, xy)[0] / expected_m, 1, places=12)
        self.assertAlmostEqual(
            decimal_reference(points, faces, xy[0]) * 1e-6 / expected_m, 1, places=12
        )

    def test_shared_edge_is_explicitly_ambiguous(self):
        points, faces = soup(box())
        self.assertTrue(np.isnan(plane_references(points, faces, [(0, 0)])[0]))
        with self.assertRaisesRegex(ValueError, "Exact edge"):
            decimal_reference(points, faces, (0, 0))


if __name__ == "__main__":
    unittest.main()
