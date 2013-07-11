import numpy as np
from numpy import linalg
import quantities as q
from syris.opticalelements import geometry as geom
from unittest import TestCase
import itertools


def get_base():
    return np.array([geom.X_AX, geom.Y_AX, geom.Z_AX]) * q.m


def get_directions(units):
    """Create directions in 3D space and apply *units*."""
    base = np.array(list(itertools.product([0, 1], [0, 1], [0, 1])))[1:]
    x_points, y_points, z_points = np.array(zip(*base))
    return np.array(zip(x_points, y_points, z_points) +
                    zip(-x_points, y_points, z_points) +
                    zip(x_points, -y_points, z_points) +
                    zip(x_points, y_points, -z_points) +
                    zip(-x_points, -y_points, -z_points)) * units


def get_vec_0():
    return np.array([1.75, -3.89, 4.7]) * q.m


class Test(TestCase):

    def test_zero_angle(self):
        zero_vec = np.zeros(3) * q.m
        self.assertEqual(geom.angle(zero_vec, zero_vec), 0 * q.deg)

        vectors = get_base()
        for vec in vectors:
            self.assertEqual(geom.angle(zero_vec, vec), 0 * q.deg)

    def test_orthogonal_angles(self):
        vectors = get_base()
        pairs = np.array([(x, y) for x in vectors for y in vectors if
                          np.array(x - y).any()]) * vectors.units
        for vec_0, vec_1 in pairs:
            self.assertEqual(geom.angle(vec_0, vec_1), 90 * q.deg)

    def test_is_normalized(self):
        norm_vec = np.array([1, 0, 0]) * q.m
        unnorm_vec = np.array([1, 1, 1]) * q.m

        self.assertTrue(geom.is_normalized(norm_vec))
        self.assertFalse(geom.is_normalized(unnorm_vec))

    def test_normalize(self):
        vec = np.array([0, 0, 0]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 0 * q.dimensionless)
        vec = np.array([1, 0, 0]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1 * q.dimensionless)
        vec = np.array([1, 1, 1]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1 * q.dimensionless)
        vec = np.array([10, 14.7, 18.75]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1 * q.dimensionless)
        vec = np.array([10, -14.7, 18.75]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1 * q.dimensionless)

    def test_length(self):
        vec = np.array([1, 1, 1]) * q.m
        self.assertAlmostEqual(geom.length(vec), np.sqrt(3) * q.m)
        vec = -vec
        self.assertAlmostEqual(geom.length(vec), np.sqrt(3) * q.m)
        vec = np.array([0, 0, 0]) * q.m
        self.assertAlmostEqual(geom.length(vec), 0 * q.m)

    def test_translate(self):
        vec_0 = get_vec_0()
        directions = get_directions(vec_0.units)

        for direction in directions:
            res_vec = geom.transform_vector(linalg.inv(
                                            geom.translate(direction)), vec_0)
            res = np.sum(res_vec - (direction + vec_0))
            self.assertAlmostEqual(res, 0)

    def test_scale(self):
        self.assertRaises(ValueError, geom.scale, np.array([0, 1, 2]))
        self.assertRaises(ValueError, geom.scale, np.array([1, -1, 2]))

        base = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        coeffs = np.array(list(itertools.product(base, base, base)))
        vec_0 = get_vec_0()

        for coeff in coeffs:
            res_vec = geom.transform_vector(
                linalg.inv(geom.scale(coeff)), vec_0)
            res = np.sum(res_vec - (coeff * vec_0))
            self.assertAlmostEqual(res, 0)

    def test_rotate(self):
        vec_1 = get_vec_0()
        normalized = geom.normalize(vec_1)

        directions = get_directions(q.dimensionless)

        for direction in directions:
            rot_axis = np.cross(direction, normalized) * q.dimensionless
            trans_mat = linalg.inv(geom.rotate(geom.angle(direction,
                                                          normalized),
                                               rot_axis))
            diff = np.sum(normalized - geom.normalize(
                          geom.transform_vector(trans_mat, direction)))
            self.assertAlmostEqual(diff, 0)
