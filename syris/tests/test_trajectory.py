'''
Created on Jul 3, 2013

@author: farago
'''
import numpy as np
import quantities as q
from scipy import interpolate as interp
from syris.geometry import Trajectory
from syris.tests import SyrisTest


class TestTrajectory(SyrisTest):

    def setUp(self):
        self.n = 100
        x = np.linspace(0, 2 * np.pi, self.n)
        # + 1 not to go below zero.
        y = 1 + np.sin(x)
        z = np.zeros(self.n)
        self.time_dist = zip(x * q.s, y * q.m)
        self.control_points = zip(x, y, z) * q.m

        self.traj = Trajectory(self.control_points, self.time_dist)

    def test_init(self):
        def test_stationary(traj):
            self.assertEqual(traj.length, 0 * q.m)
            self.assertEqual(traj.time, 0 * q.s)

        # Stationary trajectory.
        traj = Trajectory([(0, 0, 0)] * q.m)
        test_stationary(traj)

        times = np.linspace(0, 2 * np.pi, self.n)
        dist = np.sin(times)

        # Length is zero but velocities given.
        traj = Trajectory([(0, 0, 0)] * q.m, zip(times, dist))
        test_stationary(traj)

        # Length is non-zero but no velocities given.
        traj = Trajectory(self.control_points)
        test_stationary(traj)

        # Constant velocity.
        velocity = 10 * q.m / q.s
        traj = Trajectory(self.control_points, velocity=velocity)
        self.assertAlmostEqual(traj.length / velocity, traj.time)

        # Constant velocity and times and distances
        self.assertRaises(ValueError, Trajectory, self.control_points,
                          time_dist=zip(times, dist),
                          velocity=10 * q.mm / q.s)

        # Invalid velocity profile (negative distance).
        self.assertRaises(ValueError, Trajectory, self.control_points,
                          zip(times * q.s, dist * q.m))
        # Time not monotonic.
        time_dist = [(1 * q.s, 1 * q.m), (1 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points,
                          time_dist)
        time_dist = [(1 * q.s, 1 * q.m), (0 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points,
                          time_dist)

        # Negative time.
        time_dist = [(-1 * q.s, 1 * q.m), (1 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points,
                          time_dist)

    def test_length(self):
        u = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(u)
        y = np.cos(u)
        z = np.zeros(len(u))

        traj = Trajectory(zip(x, y, z) * q.m, velocity=1 * q.m / q.s)
        self.assertAlmostEqual(traj.length, 2 * np.pi * q.m, places=5)

    def test_get_maximum_dt(self):
        traj = Trajectory([(0, 0, 0)] * q.m)
        self.assertEqual(traj.get_maximum_dt(1 * q.m, 1 * q.m), None)

    def test_get_point(self):
        # Stationary trajectory
        traj = Trajectory(self.control_points)
        np.testing.assert_equal(
            traj.get_point(1 * q.s), traj.control_points[0])

        tck = interp.splprep(zip(*self.control_points), s=0)[0]

        def evaluate_point(t):
            if t > 1:
                t = 1
            return interp.splev(t, tck) * q.m

        # Create velocity profile which goes until the trajectory end.
        # We need to scale the sine amplitude in order to
        # max(sin(x)) = trajectory.length
        times = np.linspace(0, 2 * np.pi, self.n) * q.s
        # Normalize for not going below zero.
        dist = (self.traj.length + self.traj.length * np.sin(times.magnitude)) * q.m

        traj = Trajectory(self.control_points, zip(times, dist))

        for i in range(len(times)):
            np.testing.assert_almost_equal(traj.get_point(times[i]),
                                           evaluate_point(dist[i] / traj.length), decimal=4)
