import numpy as np
from numpy import linalg
import quantities as q
from syris.opticalelements import geometry as geom
from unittest import TestCase
import itertools
from testfixtures.shouldraise import ShouldRaise
from opticalelements.geometry import Trajectory


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


def get_control_points():
    return np.array([(0, 0, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (0, 0, 1),
                     (1, 1, 1)]) * q.mm


def init_trajectory(c_points=None, kwargs={}):
    if c_points is None:
        c_points = get_control_points()
    points, t_length = geom.interpolate_points(c_points, 1 * q.um)

    return Trajectory(points, t_length, **kwargs)


class Test(TestCase):

    def test_trajectory_interpolation(self):
        c_points = np.array([(0, 0, 0)]) * q.mm
        points, length = geom.interpolate_points(c_points, 1 * q.um)
        self.assertEqual(len(points), 1)
        self.assertEqual(length, 0)

        c_points = get_control_points()
        points, length = geom.interpolate_points(c_points, 1 * q.um)
        self.assertGreater(len(points), 1)
        self.assertNotEqual(length, 0)

    def test_static_trajectory(self):
        c_points = np.array([(0, 0, 0)]) * q.mm
        traj = init_trajectory(c_points)
        self.assertEqual(traj.length, 0)
        self.assertEqual(traj.time, 0)
        self.assertEqual(traj.get_distance(10 * q.s), 0)

    def test_wrong_velocities(self):
        # Too short.
        velos = [(1 * q.mm, 1 * q.mm / q.s)]
        with ShouldRaise(ValueError):
            init_trajectory(kwargs={"velocities": velos})

        # Too long.
        velos = [(1 * q.mm, 1 * q.mm / q.s), (1000 * q.mm, 10 * q.mm / q.s)]
        with ShouldRaise(ValueError):
            init_trajectory(kwargs={"velocities": velos})

    def test_trajectory(self):
        c_points = get_control_points()
        points, length = geom.interpolate_points(c_points, 1 * q.um)
        s_0 = 1 * q.mm
        s_1 = 3 * q.mm
        s_2 = length - s_1 - s_0
        v_0 = 10 * q.mm / q.s
        v_1 = 10 * q.mm / q.s
        v_2 = 4 * q.mm / q.s
        dist_velos = [(s_0, v_0), (s_1, v_1), (s_2, v_2)]
        traj = Trajectory(points, length, dist_velos)

        self.assertGreater(traj.length, 0)
        self.assertNotEqual(traj.time, 0)
        self.assertAlmostEqual(traj.length, traj.get_distance(traj.time))
        self.assertAlmostEqual(traj.length,
                               traj.get_distance(traj.time + 1 * q.s))

        orig_times = np.array([velo[0].magnitude
                               for velo in traj.time_profile]) * \
            traj.time_profile[0][0].units
        total_time = 0 * orig_times[0].units
        times = np.zeros(orig_times.shape) * orig_times.units
        for i in range(len(times)):
            times[i] = total_time + orig_times[i] / 2
            total_time += orig_times[i]

        t_0, t_1, t_2 = times
        accel = v_0 / orig_times[0]
        s_0 = 0.5 * accel * t_0 ** 2
        s_0_full = 0.5 * v_0 * orig_times[0]
        s_1 = s_0_full + v_1 * orig_times[1] / 2
        s_1_full = s_0_full + v_1 * orig_times[1]
        decel = (v_1 - v_2) / orig_times[2]
        s_2 = s_1_full + v_1 * \
            orig_times[2] / 2 - 0.5 * decel * (orig_times[2] / 2) ** 2
        s_2_full = s_1_full + v_1 * \
            orig_times[2] - 0.5 * (v_1 - v_2) * orig_times[2]

        self.assertAlmostEqual(traj.get_distance(t_0), s_0)
        self.assertAlmostEqual(s_0_full, traj.get_distance(orig_times[0]))
        self.assertAlmostEqual(traj.get_distance(t_1), s_1)
        self.assertAlmostEqual(s_1_full,
                               traj.get_distance(orig_times[0] +
                                                 orig_times[1]))
        self.assertAlmostEqual(s_2, traj.get_distance(t_2))
        self.assertAlmostEqual(s_2_full, traj.get_distance(np.sum(orig_times)))

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
        self.assertEqual(geom.length(geom.normalize(vec)), 0)
        vec = np.array([1, 0, 0]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1)
        vec = np.array([1, 1, 1]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1)
        vec = np.array([10, 14.7, 18.75]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1)
        vec = np.array([10, -14.7, 18.75]) * q.m
        self.assertEqual(geom.length(geom.normalize(vec)), 1)

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
        with ShouldRaise(ValueError):
            geom.scale(np.array([0, 1, 2]))
        with ShouldRaise(ValueError):
            geom.scale(np.array([1, -1, 2]))

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
            rot_axis = np.cross(direction, vec_1)
            trans_mat = linalg.inv(geom.rotate(geom.angle(direction, vec_1),
                                               rot_axis))
            diff = np.sum(normalized - geom.normalize(
                          geom.transform_vector(trans_mat, direction)))
            self.assertAlmostEqual(diff, 0)
