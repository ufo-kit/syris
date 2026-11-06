"""Normal transforms, including the unit-bearing PyVista loading path.

These tests need no CUDA or OpenCL device.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import quantities as q

from syris import config as cfg
from syris.bodies.mesh import Mesh
from syris.geometry import Trajectory


class TestMeshNormals(unittest.TestCase):
    def test_supplied_normals_preserve_units_and_follow_affine_transform(self):
        for double in (False, True):
            for unit in (None, q.dimensionless, q.um):
                with (
                    self.subTest(double=double, unit=unit),
                    patch.multiple(cfg, PRECISION=cfg.Precision(double), UNIT=q.um),
                ):
                    dtype = cfg.PRECISION.np_float
                    vertices = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]], dtype=dtype)
                    supplied = np.array([[-1, -1, 1]], dtype=dtype)
                    if unit is not None:
                        supplied = supplied * unit
                    original = supplied.copy()
                    mesh = Mesh(
                        vertices.T * q.um,
                        Trajectory([(0, 0, 0)] * q.um),
                        center=None,
                        normals=supplied,
                        use_normals=True,
                    )
                    # Rotation followed by unequal scaling: rotating normals
                    # as vectors would fail the orthogonality check below.
                    mesh.transform_matrix = np.array(
                        [[0, -3, 0, 0], [2, 0, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1]],
                        dtype=dtype,
                    )
                    mesh.transform()
                    actual = mesh.normals
                    if unit is not None:
                        self.assertIsInstance(actual, q.Quantity)
                        self.assertEqual(actual.units, supplied.units)
                    np.testing.assert_allclose(np.asarray(actual), [[1 / 3, -0.5, 2]], rtol=1e-7)
                    for edge in mesh.vectors:
                        np.testing.assert_allclose(
                            np.sum(np.asarray(actual) * edge.magnitude, axis=1), 0, atol=1e-7
                        )
                    np.testing.assert_array_equal(supplied, original)

    def test_pyvista_loader_normals_preserve_units_after_rotation(self):
        with patch.multiple(cfg, PRECISION=cfg.Precision(), UNIT=q.um):
            surface = pv.PolyData(
                np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([3, 0, 1, 2]),
            )
            mesh = Mesh.from_file(
                surface, Trajectory([(0, 0, 0)] * q.um), center=None, unit=q.um, use_normals=True
            )
            self.assertIsInstance(mesh.normals, q.Quantity)
            np.testing.assert_allclose(mesh.normals.rescale(q.um).magnitude, [[1, 0, 0]])
            mesh.transform_matrix = np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            mesh.transform()
            np.testing.assert_allclose(mesh.normals.rescale(q.um).magnitude, [[0, 1, 0]])
