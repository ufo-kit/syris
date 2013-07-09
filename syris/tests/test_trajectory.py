'''
Created on Jul 3, 2013

@author: farago
'''
import numpy as np
import quantities as q
from syris.opticalelements import geometry as geom
from unittest import TestCase
from syris.opticalelements.geometry import Trajectory
from testfixtures.shouldraise import ShouldRaise


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
        tmp = traj.get_point(0 * q.s) - c_points
        self.assertEqual(np.sum(tmp), 0)

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
