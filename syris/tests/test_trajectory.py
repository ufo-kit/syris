"""
Created on Jul 3, 2013

@author: farago
"""
import numpy as np
import quantities as q
from scipy import interpolate as interp
from syris.geometry import get_rotation_displacement, Trajectory
from syris.tests import SyrisTest


def make_circle(n=128):
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros(n)

    return list(zip(x, y, z)) * q.mm


def make_square(n=128):
    t = np.linspace(0, 2 * np.pi, n)
    x = np.zeros(n)
    y = t ** 2
    z = np.zeros(n)

    return list(zip(x, y, z)) * q.mm


def create_maxima_testing_data():
    n = 128
    points = make_circle(n=n) * 1e3
    ps = 10 * q.mm
    furthest = 1 * q.m

    tr = Trajectory(points, ps, furthest, velocity=1 * q.m / q.s)
    duration = tr.time.simplified.magnitude
    t = np.linspace(0, 2 * duration, n) * q.s
    # Sine-like velocity profile
    s = 0.5 * tr.length * (np.sin(t.magnitude / duration * 2 * np.pi) + 1)
    time_dist = list(zip(t, s))
    # Use small num_points to make sure we move more than 1 px
    tr = Trajectory(points, ps, furthest, time_dist=time_dist, num_points=n)
    distances = tr.get_distances()
    dtck = interp.splprep(distances, u=tr.parameter, s=0, k=1)[0]

    return tr, dtck, ps, t


class TestTrajectory(SyrisTest):
    def setUp(self):
        self.n = 100
        x = np.linspace(0, 2 * np.pi, self.n)
        # + 1 not to go below zero.
        y = 1 + np.sin(x)
        z = np.zeros(self.n)
        self.time_dist = list(zip(x * q.s, y * q.m))
        self.control_points = list(zip(x, y, z)) * q.m

        self.traj = Trajectory(
            self.control_points,
            pixel_size=1 * q.mm,
            furthest_point=1 * q.mm,
            time_dist=self.time_dist,
        )

    def test_init(self):
        def _test_stationary(traj):
            self.assertEqual(traj.length, 0 * q.m)
            self.assertEqual(traj.time, 0 * q.s)

        # Stationary trajectory.
        traj = Trajectory([(0, 0, 0)] * q.m)
        _test_stationary(traj)

        times = np.linspace(0, 2 * np.pi, self.n)
        dist = np.sin(times)

        # Length is zero but velocities given.
        traj = Trajectory([(0, 0, 0)] * q.m, list(zip(times, dist)))
        _test_stationary(traj)

        # Length is non-zero but no velocities given.
        traj = Trajectory(self.control_points)
        _test_stationary(traj)

        # Constant velocity.
        velocity = 10 * q.m / q.s
        traj = Trajectory(self.control_points, velocity=velocity)
        self.assertAlmostEqual(traj.length / velocity, traj.time)

        # Constant velocity and times and distances
        self.assertRaises(
            ValueError,
            Trajectory,
            self.control_points,
            time_dist=list(zip(times, dist)),
            velocity=10 * q.mm / q.s,
        )

        # Invalid velocity profile (negative distance).
        self.assertRaises(
            ValueError,
            Trajectory,
            self.control_points,
            1 * q.m,
            1 * q.m,
            list(zip(times * q.s, dist * q.m)),
        )
        # Time not monotonic.
        time_dist = [(1 * q.s, 1 * q.m), (1 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points, 1 * q.m, 1 * q.m, time_dist)
        time_dist = [(1 * q.s, 1 * q.m), (0 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points, 1 * q.m, 1 * q.m, time_dist)

        # Negative time.
        time_dist = [(-1 * q.s, 1 * q.m), (1 * q.s, 1 * q.m)]
        self.assertRaises(ValueError, Trajectory, self.control_points, 1 * q.m, 1 * q.m, time_dist)

    def test_stationary(self):
        traj = Trajectory(self.control_points)
        self.assertTrue(traj.stationary)
        traj = Trajectory(self.control_points, time_dist=self.time_dist)
        self.assertFalse(traj.stationary)
        traj = Trajectory(self.control_points, velocity=1 * q.mm / q.s)
        self.assertFalse(traj.stationary)

    def test_bind(self):
        traj = Trajectory(self.control_points, time_dist=self.time_dist)
        self.assertFalse(traj.bound)
        traj.bind(pixel_size=1 * q.m, furthest_point=1 * q.um)
        self.assertTrue(traj.bound)

        # Trajectory with no furthest point must work too
        traj.bind(pixel_size=1 * q.m)
        self.assertTrue(traj.bound)

        # Binding stationary trajectory must be possible
        traj = Trajectory([(0, 0, 0)] * q.m)
        traj.bind(pixel_size=100 * q.mm)
        self.assertTrue(traj.bound)

    def test_length(self):
        u = np.linspace(0, 2 * np.pi, 100)
        x = np.sin(u)
        y = np.cos(u)
        z = np.zeros(len(u))

        traj = Trajectory(
            list(zip(x, y, z)) * q.m,
            velocity=1 * q.m / q.s,
            pixel_size=1 * q.m,
            furthest_point=1 * q.m,
        )
        self.assertAlmostEqual(traj.length, 2 * np.pi * q.m, places=5)

    def test_get_point(self):
        # Stationary trajectory
        traj = Trajectory(self.control_points, pixel_size=1 * q.m, furthest_point=1 * q.m)
        np.testing.assert_equal(traj.get_point(1 * q.s), traj.control_points[0])

        tck = interp.splprep(list(zip(*self.control_points)), s=0)[0]

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

        traj = Trajectory(
            self.control_points,
            pixel_size=1 * q.m,
            furthest_point=1 * q.m,
            time_dist=list(zip(times, dist)),
        )

        for i in range(len(times)):
            np.testing.assert_almost_equal(
                traj.get_point(times[i]), evaluate_point(dist[i] / traj.length), decimal=4
            )

    def test_get_next_time(self):
        """Very small rotation circle but large object extent."""

        def pr(v, decimals=2):
            return np.round(v, decimals)

        points = make_circle(n=128) * 1e-3
        x, y, z = list(zip(*points))
        furthest = 1 * q.mm
        ps = 10 * q.um
        tr = Trajectory(points, ps, furthest, velocity=1 * q.mm / q.s)

        t_0 = 0 * q.s
        t_1 = 0 * q.s
        max_diff = 0 * q.m

        while t_1 != np.inf * q.s:
            t_1 = tr.get_next_time(t_0)
            p_0 = tr.get_point(t_0)
            p_1 = tr.get_point(t_1)
            d_0 = tr.get_direction(t_0, norm=True)
            d_1 = tr.get_direction(t_1, norm=True)
            dp = np.abs(p_1 - p_0)
            dd = np.abs(d_1 - d_0) * furthest
            total = dp + dd
            ds = max(total)
            if ds > max_diff:
                max_diff = ds
            t_0 = t_1

        max_diff = max_diff.rescale(q.um).magnitude
        ps = ps.rescale(q.um).magnitude
        np.testing.assert_almost_equal(ps, max_diff, decimal=2)

    def test_get_distances(self):
        """Compare analytically computed distances with the ones obtained from trajectory."""
        points = make_circle(n=128)
        x, y, z = list(zip(*points))
        furthest = 3 * q.mm
        ps = 10 * q.um
        tr = Trajectory(points, ps, furthest, velocity=1 * q.mm / q.s)

        d_points = np.abs(tr.points[:, 0][:, np.newaxis] - tr.points)
        t = np.linspace(0, 2 * np.pi, tr.points.shape[1])
        x, y, z = list(zip(*make_circle(n=tr.points.shape[1])))
        dx = -np.sin(t)
        dy = np.cos(t)
        dz = z
        derivatives = np.array(list(zip(dx, dy, dz))).T.copy()
        d_derivatives = get_rotation_displacement(derivatives[:, 0], derivatives, furthest)
        distances = (d_points + d_derivatives).simplified.magnitude

        np.testing.assert_almost_equal(distances, tr.get_distances())

    def test_get_maximum_du(self):
        """Test that we don't move by more than the pixel size between two consecutive parameter
        values.
        """
        tr, dtck, ps, times = create_maxima_testing_data()

        max_du = tr.get_maximum_du(distance=ps)
        du = np.gradient(tr.parameter)
        # Make sure we are actually making too large steps with the original interpolation
        assert max_du < np.max(du)
        u = np.arange(0, 1, max_du)
        du = np.gradient(u)
        distance = np.max(np.abs(np.array(interp.splev(u, dtck, der=1)) * du)) * 1e3

        np.testing.assert_almost_equal(ps.rescale(q.mm).magnitude, distance, decimal=2)

    def test_get_maximum_dt(self):
        """Test that we don't move by more than the pixel size between two consecutive times."""
        # Stationary trajectory
        traj = Trajectory([(0, 0, 0)] * q.m, pixel_size=1 * q.m, furthest_point=1 * q.m)
        self.assertEqual(traj.get_maximum_dt(), None)

        # Moving trajectory
        tr, dtck, ps, times = create_maxima_testing_data()
        max_dt = tr.get_maximum_dt(distance=ps).simplified.magnitude
        # Make sure we are actually making too large steps with the original interpolation
        assert max_dt < np.max(np.gradient(tr.times.simplified.magnitude))
        times = np.arange(times[0].simplified.magnitude, times[-1].simplified.magnitude, max_dt)
        u = interp.splev(times, tr.time_tck) / tr.length.simplified.magnitude
        du = np.gradient(u)
        distance = np.max(np.abs(np.array(interp.splev(u, dtck, der=1)) * du)) * 1e3

        np.testing.assert_almost_equal(ps.rescale(q.mm).magnitude, distance, decimal=1)
