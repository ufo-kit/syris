import numpy as np
import quantities as q
import syris
import syris.config as cfg
from syris import geometry as geom
from syris.geometry import Trajectory
from syris.bodies.base import CompositeBody
from syris.bodies.isosurfaces import MetaBall
from syris.bodies.mesh import make_cube, Mesh
from syris.bodies.simple import StaticBody
from syris.imageprocessing import crop, pad, rescale
from syris.materials import Material
from syris.tests import SyrisTest, opencl, slow
import itertools
from syris.tests.graphics_util import get_linear_points


def get_control_points():
    return np.array([(0, 0, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (0, 0, 1),
                     (1, 1, 1)]) * q.mm


@opencl
def test_simple():
    syris.init(device_index=0)
    n = 8
    ps = 1 * q.um
    thickness = np.arange(n ** 2).reshape(n, n).astype(cfg.PRECISION.np_float) * q.m
    go = StaticBody(thickness, ps)

    # Same
    projection = go.project((n, n), ps).get()
    np.testing.assert_almost_equal(thickness.magnitude, projection)

    # Cropped upsampled
    shape = (n, n)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, ps / 2).get()
    gt = rescale(crop(thickness.magnitude, (0, 0, n / 2, n / 2)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Cropped downsampled
    shape = (n / 4, n / 4)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, 2 * ps).get()
    gt = rescale(crop(thickness.magnitude, (0, 0, n / 2, n / 2)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Padded upsampled
    shape = (4 * n, 4 * n)
    projection = go.project(shape, ps / 2, offset=(-n / 2, -n / 2) * ps).get()
    gt = rescale(pad(thickness.magnitude, (n / 2, n / 2, 2 * n, 2 * n)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Padded downsampled
    shape = (n, n)
    projection = go.project(shape, 2 * ps, offset=(-n / 2, -n / 2) * ps).get()
    gt = rescale(pad(thickness.magnitude, (4, 4, 2 * n, 2 * n)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Crop and pad and upsample
    def crop_pad_rescale(ss):
        shape = (n, n)
        target_ps = ps / ss
        fov = shape * ps
        offset = (2, -2) * q.um
        shape = (int(n / 2 * ss), int(ss * 3 * n / 2))
        projection = go.project(shape, target_ps, offset=offset).get()
        gt = rescale(pad(thickness[n/4:3*n/4, :], (0, n / 4, n / 2, 3 * n / 2)), shape).get()
        np.testing.assert_almost_equal(gt, projection)

    crop_pad_rescale(2)
    crop_pad_rescale(0.5)


class TestBodies(SyrisTest):

    def setUp(self):
        syris.init(device_index=0)
        self.pixel_size = 1 * q.um

        control_points = get_linear_points(geom.X, start=(1, 1, 1))
        traj = Trajectory(control_points, velocity=1 * q.mm / q.s)
        self.metaball = MetaBall(traj, 1 * q.mm)
        self.metaball_2 = MetaBall(Trajectory(get_linear_points(geom.Z)), 2 * q.mm)
        self.composite = CompositeBody(traj, bodies=[self.metaball, self.metaball_2])

    def _get_moved_bounding_box(self, body, angle):
        body.translate((1, 0, 0) * q.mm)
        body.rotate(angle, np.array((0, 0, 1)))
        body.translate((1, 0, 0) * q.mm)

        base = -2 * body.radius.magnitude, 2 * body.radius.magnitude
        transformed = []
        for point in list(itertools.product(base, base, base)):
            transformed.append(geom.transform_vector(body.transform_matrix,
                                                     point * body.radius.units).
                                                     simplified.magnitude)

        return transformed * q.m

    def test_metaball(self):
        self.assertEqual(self.metaball.radius, 1 * q.mm)
        np.testing.assert_almost_equal(self.metaball.center,
                                       np.array([1, 1, 1]) * q.mm)

    def test_metaball_bounding_box(self):
        """Bounding box moves along with its body."""
        mb = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 0.5 * q.mm)
        transformed = self._get_moved_bounding_box(mb, 90 * q.deg)

        np.testing.assert_almost_equal(transformed, mb.bounding_box.points)

    def test_composite_subbodies(self):
        m_1 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 1 * q.mm)
        m_2 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 2 * q.mm)
        m_3 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 3 * q.mm)
        m_4 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 4 * q.mm)
        m_5 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 5 * q.mm)
        m_6 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 6 * q.mm)
        m_7 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 7 * q.mm)

        c_1 = CompositeBody(Trajectory([(0, 0, 0)] * q.mm),
                              bodies=[m_1, m_2])
        c_2 = CompositeBody(Trajectory([(0, 0, 0)] * q.mm),
                              bodies=[m_3])
        c_3 = CompositeBody(Trajectory([(0, 0, 0)] * q.mm),
                              bodies=[c_1, c_2])
        c_4 = CompositeBody(Trajectory([(0, 0, 0)] * q.mm),
                              bodies=[m_4, m_5])
        c_5 = CompositeBody(Trajectory([(0, 0, 0)] * q.mm),
                              bodies=[c_3, c_4, m_6, m_7])

        # Empty composit body
        CompositeBody(Trajectory([(0, 0, 0)] * q.mm))

        # Empty composite body.
        c_empty = CompositeBody(Trajectory([(0, 0, 0)] * q.mm))
        self.assertEqual(c_empty.direct_primitive_bodies, [])

        # Test direct subbodies
        self.assertEqual(c_5.direct_primitive_bodies, [m_6, m_7])
        self.assertEqual(c_3.direct_primitive_bodies, [])

        # Add self.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(c_5)

        # Add already contained primitive body.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(m_1)

        # Add already contained composite body.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(c_2)

        # Test all subbodies.
        self.assertEqual(set(c_3.all_bodies),
                         set([m_1, m_2, m_3, c_1, c_2, c_3]))
        self.assertEqual(set(c_1.all_bodies), set([c_1, m_1, m_2]))
        self.assertEqual(c_empty.all_bodies, (c_empty,))

    def test_composite_bounding_box(self):
        mb_0 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 0.5 * q.mm)
        mb_1 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 1.5 * q.mm)
        composite = CompositeBody(Trajectory([(0, 0, 0)] * q.mm,),
                                    bodies=[mb_0, mb_1])

        transformed_0 = self._get_moved_bounding_box(mb_0, 90 * q.deg)
        transformed_1 = self._get_moved_bounding_box(mb_1, -90 * q.deg)

        def get_concatenated(t_0, t_1, index):
            return np.concatenate((list(zip(*t_0))[index], list(zip(*t_1))[index])) * \
                t_0[0].units

        x_points = get_concatenated(transformed_0, transformed_1, 0)
        y_points = get_concatenated(transformed_0, transformed_1, 1)
        z_points = get_concatenated(transformed_0, transformed_1, 2)

        x_min_max = min(x_points), max(x_points)
        y_min_max = min(y_points), max(y_points)
        z_min_max = min(z_points), max(z_points)

        ground_truth = np.array(list(
                                itertools.product(x_min_max,
                                                  y_min_max, z_min_max))) * q.m

        np.testing.assert_almost_equal(ground_truth,
                                       composite.bounding_box.points)

    def test_composite_furthest_point(self):
        mb_0 = MetaBall(Trajectory([(0, 0, 0)] * q.m), 1 * q.m)
        mb_1 = MetaBall(Trajectory([(0, 0, 0)] * q.m), 2 * q.m)
        composite = CompositeBody(Trajectory([(0, 0, 0)] * q.m), bodies=[mb_0, mb_1])

        # Metaball's furthest point is twice the radius
        self.assertAlmostEqual(4 * q.m, composite.furthest_point)

    def test_save_transformation_matrix(self):
        old = self.composite.transform_matrix
        self.composite.bind_trajectory(self.pixel_size)
        self.composite.save_transformation_matrices()

        self.composite.move(1 * q.s)
        self.composite.restore_transformation_matrices()

        np.testing.assert_equal(old, self.composite.transform_matrix)

    def test_get_distance(self):
        n = 100
        ps = 100 * q.mm
        p = np.linspace(0, np.pi, n)
        sin = np.sin(p)
        cos = np.cos(p)
        zeros = np.zeros(n)

        # Simple body
        # -----------
        traj = Trajectory(list(zip(cos, sin, zeros)) * q.m, velocity=1 * q.m / q.s)
        ball = MetaBall(traj, .5 * q.m)
        ball.bind_trajectory(ps)
        dist = ball.get_distance(0 * q.s, ball.trajectory.time)
        # Maximum along the x axis where the body travels 2 m by translation and rotates by 180
        # degrees compared to position at t0, so the rotational displacement is 2 * furthest point,
        # in this case 2 m, so altogether 4 m
        self.assertAlmostEqual(dist.simplified.magnitude, 4)

        # Composite body
        # --------------
        traj_m = Trajectory([(0, 0, 0)] * q.m)
        ball = MetaBall(traj_m, .5 * q.m)
        comp = CompositeBody(traj, bodies=[ball])
        comp.bind_trajectory(ps)

        d = comp.get_distance(0 * q.s, comp.time).simplified.magnitude
        # 2 by translation and 180 degrees means 2 * furthest
        gt = 2 * ball.furthest_point.simplified.magnitude + 2
        self.assertAlmostEqual(gt, d, places=4)

        d = comp.get_distance(0 * q.s, comp.time / 2).simplified.magnitude
        # 1 by translation by either x or y and sqrt(2) * furthest by rotation
        gt = 1 + comp.furthest_point.simplified.magnitude * np.sqrt(2)
        self.assertAlmostEqual(gt, d, places=4)

    @slow
    def test_get_next_time(self):
        n = 100
        ps = 100 * q.mm
        psm = ps.simplified.magnitude
        p = np.linspace(0, np.pi, n)
        sin = np.sin(p) * 1e-3
        cos = np.cos(p) * 1e-3
        zeros = np.zeros(n)

        traj_m_0 = Trajectory(list(zip(p * 1e-3, zeros, zeros)) * q.m, velocity=1 * q.mm / q.s)
        traj_m_1 = Trajectory([(0, 0, 0)] * q.m)
        ball_0 = MetaBall(traj_m_0, .5 * q.m)
        ball_1 = MetaBall(traj_m_1, .5 * q.m)
        traj = Trajectory(list(zip(cos, sin, zeros)) * q.m, velocity=1 * q.mm / q.s)
        comp = CompositeBody(traj, bodies=[ball_0, ball_1])
        dt = comp.get_maximum_dt(ps)

        # Normal trajectories
        # Test the beginning, middle and end because of time complexity
        for t_0 in [0 * q.s, comp.time / 2, comp.time - 10 * dt]:
            t_1 = comp.get_next_time(t_0, ps)

            d = comp.get_distance(t_0, t_1).simplified.magnitude
            np.testing.assert_almost_equal(psm, d)

        # Trajectories which sum up to no movement
        traj_m = Trajectory(list(zip(zeros, -p, zeros)) * q.m, velocity=1 * q.mm / q.s)
        ball = MetaBall(traj_m, .5 * q.m)
        traj = Trajectory(list(zip(p, zeros, zeros)) * q.m, velocity=1 * q.mm / q.s)
        comp = CompositeBody(traj, bodies=[ball])

        self.assertEqual(np.inf * q.s, comp.get_next_time(0 * q.s, ps))

    def test_static_composite(self):
        ps = 1 * q.um
        traj = Trajectory([(0, 0, 0)] * q.m)
        mb_0 = MetaBall(traj, 10 * q.um)
        mb_1 = MetaBall(traj, 10 * q.um)
        comp = CompositeBody(traj, bodies=[mb_0, mb_1])
        self.assertEqual(np.inf * q.s, comp.get_next_time(0 * q.s, ps))

    @opencl
    @slow
    def test_project_composite(self):
        n = 64
        shape = (n, n)
        ps = 1 * q.um
        x = np.linspace(0, n, num=10)
        y = z = np.zeros(x.shape)
        traj_x = Trajectory(list(zip(x, y, z)) * ps, velocity=ps / q.s)
        traj_y = Trajectory(list(zip(y, x, z)) * ps, velocity=ps / q.s)
        traj_xy = Trajectory(list(zip(n - x, x, z)) * ps, velocity=ps / q.s)
        mb = MetaBall(traj_x, n * ps / 16)
        cube = make_cube() / q.m * 16 * ps / 4
        mesh = Mesh(cube, traj_xy)
        composite = CompositeBody(traj_y, bodies=[mb, mesh])
        composite.bind_trajectory(ps)

        composite.move(n / 2 * q.s)
        p = composite.project(shape, ps).get()
        composite.clear_transformation()

        # Compute
        composite.move(n / 2 * q.s)
        p_separate = (mb.project(shape, ps) + mesh.project(shape, ps)).get()

        np.testing.assert_almost_equal(p, p_separate)

    def _run_parallel_or_opposite(self, sgn, gt):
        n = 64
        ps = 1 * q.um
        y = np.linspace(0, sgn * n, num=10)
        x = z = np.zeros(y.shape)
        traj = Trajectory(list(zip(x, y, z)) * ps, velocity=ps / q.s, pixel_size=ps)
        mb = MetaBall(traj, n * ps / 16, orientation=geom.Y_AX)

        mb.move(0 * q.s)
        np.testing.assert_almost_equal(gt, mb.transform_matrix)

    def test_opposite_trajectory(self):
        """Body's orientation is exactly opposite of the trajectory direction."""
        gt = geom.rotate(180 * q.deg, geom.Z_AX)
        self._run_parallel_or_opposite(-1, gt)

    def test_parallel_trajectory(self):
        """Body's orientation is exactly same as the trajectory direction."""
        gt = np.identity(4)
        self._run_parallel_or_opposite(1, gt)
