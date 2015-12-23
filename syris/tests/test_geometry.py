import numpy as np
from numpy import linalg
import quantities as q
from syris import geometry as geom
import itertools
from syris.geometry import BoundingBox
from syris.tests import SyrisTest


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


class TestGeometry(SyrisTest):

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
        norm_vec = np.array([1, 0, 0]) * q.dimensionless
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
            res_vec = geom.transform_vector(geom.translate(direction), vec_0)
            res = np.sum(res_vec - (direction + vec_0))
            self.assertAlmostEqual(res, 0)

    def test_scale(self):
        self.assertRaises(ValueError, geom.scale, np.array([0, 1, 2]))
        self.assertRaises(ValueError, geom.scale, np.array([1, -1, 2]))

        base = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        coeffs = np.array(list(itertools.product(base, base, base)))
        vec_0 = get_vec_0()

        for coeff in coeffs:
            res_vec = geom.transform_vector(geom.scale(coeff), vec_0)
            res = np.sum(res_vec - (coeff * vec_0))
            self.assertAlmostEqual(res, 0)

    def test_rotate(self):
        vec_1 = get_vec_0()
        normalized = geom.normalize(vec_1)

        directions = get_directions(q.dimensionless)

        for direction in directions:
            rot_axis = np.cross(direction, normalized) * q.dimensionless
            trans_mat = geom.rotate(geom.angle(direction, normalized), rot_axis)
            diff = np.sum(normalized - geom.normalize(
                          geom.transform_vector(trans_mat, direction)))
            self.assertAlmostEqual(diff, 0)

    def test_overlap(self):
        self.assertTrue(geom.overlap((0, 2), (1, 3)))
        self.assertFalse(geom.overlap((0, 2), (2, 3)))
        self.assertFalse(geom.overlap((0, 2), (-1, 0)))
        self.assertFalse(geom.overlap((0, 2), (-10, -5)))
        self.assertFalse(geom.overlap((0, 2), (3, 10)))
        self.assertTrue(geom.overlap((0, 1), (0, 1)))

    def test_bounding_box_overlap(self):
        def test(base, ground_truth):
            b_1 = BoundingBox(list(itertools.product(base, base, base)) *
                              q.m)
            self.assertEqual(b_0.overlaps(b_1), ground_truth)

        base_0 = -1, 1
        b_0 = BoundingBox(list(itertools.product(base_0, base_0, base_0)) *
                          q.m)
        test((1, 2), False)
        test((-10, -5), False)
        test((0, 2), True)
        test((-1, 0), True)
        test((-1, 1), True)

    def test_make_points(self):
        x_ends = [0, 1] * q.mm
        y_ends = [2, 3] * q.mm
        z_ends = [4, 5] * q.mm

        gt = list(itertools.product(x_ends.magnitude, y_ends.magnitude, z_ends.magnitude)) * q.mm
        points = geom.make_points(x_ends, y_ends, z_ends)
        np.testing.assert_equal(gt, points)

    def test_get_rotation_displacement(self):
        d_0 = np.array((0, 2, 0))
        d_1 = np.array((1, 0, 0))
        length = 5 * q.m
        # 90 degrees around z moves both x and y components by the *length*
        gt = np.array((length, length, 0)) * q.m

        displ = geom.get_rotation_displacement(d_0, d_1, length)
        np.testing.assert_equal(gt, displ)
