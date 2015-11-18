import numpy as np
import quantities as q
import syris
import syris.config as cfg
from syris import geometry as geom
from syris.geometry import Trajectory
from syris.graphicalobjects import MetaBall, CompositeObject, SimpleGraphicalObject
from syris.imageprocessing import crop, pad, rescale
from syris.materials import Material
from syris.tests import SyrisTest, slow
import itertools
from numpy import linalg
from syris.tests.graphics_util import get_linear_points


def get_control_points():
    return np.array([(0, 0, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (0, 0, 1),
                     (1, 1, 1)]) * q.mm


def check_distances(graphical_object, distance, decimal_points=3):
    t_0 = 0 * q.s
    max_distances = []
    while t_0 <= graphical_object.trajectory.time:
        t_1 = graphical_object.get_next_time(t_0, distance)
        if t_1 is None:
            break
        if t_0 is not None and t_1 is not None:
            diff = np.abs(graphical_object.trajectory.get_point(t_1) -
                          graphical_object.trajectory.get_point(t_0))
            max_distances.append(np.max(diff).magnitude)
        t_0 = t_1

    distance = distance.simplified.magnitude
    np.testing.assert_almost_equal(max_distances, distance, decimal=decimal_points)


def test_simple():
    syris.init()
    n = 8
    ps = 1 * q.um
    thickness = np.arange(n ** 2).reshape(n, n).astype(cfg.PRECISION.np_float) * q.m
    go = SimpleGraphicalObject(thickness, ps)

    # Same
    projection = go.project((n, n), ps).get()
    np.testing.assert_almost_equal(thickness.magnitude, projection)

    # Cropped upsampled
    shape = (n, n)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, ps / 2).get()
    gt = rescale(crop(thickness.magnitude, (2, 2, n / 2, n / 2)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Cropped downsampled
    shape = (n / 4, n / 4)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, 2 * ps).get()
    gt = rescale(crop(thickness.magnitude, (2, 2, n / 2, n / 2)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Padded upsampled
    shape = (4 * n, 4 * n)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, ps / 2).get()
    gt = rescale(pad(thickness.magnitude, (4, 4, 2 * n, 2 * n)), shape).get()
    np.testing.assert_almost_equal(gt, projection)

    # Padded downsampled
    shape = (n, n)
    gt = rescale(thickness.magnitude, shape).get()
    projection = go.project(shape, 2 * ps).get()
    gt = rescale(pad(thickness.magnitude, (4, 4, 2 * n, 2 * n)), shape).get()
    np.testing.assert_almost_equal(gt, projection)


class TestGraphicalObjects(SyrisTest):

    def setUp(self):
        syris.init()
        self.pixel_size = 1 * q.um

        control_points = get_linear_points(geom.X, start=(1, 1, 1))

        traj = Trajectory(control_points, velocity=1 * q.mm / q.s)
        self.metaball = MetaBall(traj, 1 * q.mm)

        self.metaball_2 = MetaBall(
            Trajectory(get_linear_points(geom.Z)), 2 * q.mm)

        self.composite = CompositeObject(traj,
                                         gr_objects=[self.metaball,
                                                     self.metaball_2])

    def _get_moved_bounding_box(self, obj, angle):
        obj.translate((1, 0, 0) * q.mm)
        obj.rotate(angle, np.array((0, 0, 1)))
        obj.translate((1, 0, 0) * q.mm)

        base = -2 * obj.radius.magnitude, 2 * obj.radius.magnitude
        transformed = []
        for point in list(itertools.product(base, base, base)):
            transformed.append(geom.transform_vector(linalg.inv(
                obj.transform_matrix), point * obj.radius.units).
                simplified.magnitude)

        return transformed * q.m

    def test_metaball(self):
        self.assertEqual(self.metaball.radius, 1 * q.mm)
        np.testing.assert_almost_equal(self.metaball.center,
                                       np.array([1, 1, 1]) * q.mm)

    def test_metaball_bounding_box(self):
        """Bounding box moves along with its object."""
        mb = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 0.5 * q.mm)
        transformed = self._get_moved_bounding_box(mb, 90 * q.deg)

        np.testing.assert_almost_equal(transformed, mb.bounding_box.points)

    def test_composite_subobjects(self):
        m_1 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 1 * q.mm)
        m_2 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 2 * q.mm)
        m_3 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 3 * q.mm)
        m_4 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 4 * q.mm)
        m_5 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 5 * q.mm)
        m_6 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 6 * q.mm)
        m_7 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 7 * q.mm)

        c_1 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_1, m_2])
        c_2 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_3])
        c_3 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[c_1, c_2])
        c_4 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_4, m_5])
        c_5 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[c_3, c_4, m_6, m_7])

        # empty object
        CompositeObject(Trajectory([(0, 0, 0)] * q.mm))

        # Empty composite object.
        c_empty = CompositeObject(Trajectory([(0, 0, 0)] * q.mm))
        self.assertEqual(c_empty.direct_primitive_objects, [])

        # Test direct subobjects
        self.assertEqual(c_5.direct_primitive_objects, [m_6, m_7])
        self.assertEqual(c_3.direct_primitive_objects, [])

        # Add self.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(c_5)
        self.assertEqual("Cannot add self", ctx.exception.message)

        # Add already contained primitive object.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(m_1)
        self.assertTrue(ctx.exception.message.endswith("already contained"))

        # Add already contained composite object.
        with self.assertRaises(ValueError) as ctx:
            c_5.add(c_2)
        self.assertTrue(ctx.exception.message.endswith("already contained"))

        # Test all subobjects.
        self.assertEqual(set(c_3.all_objects),
                         set([m_1, m_2, m_3, c_1, c_2, c_3]))
        self.assertEqual(set(c_1.all_objects), set([c_1, m_1, m_2]))
        self.assertEqual(c_empty.all_objects, (c_empty,))

    def test_composite_bounding_box(self):
        mb_0 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 0.5 * q.mm)
        mb_1 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 1.5 * q.mm)
        composite = CompositeObject(Trajectory([(0, 0, 0)] * q.mm,),
                                    gr_objects=[mb_0, mb_1])

        transformed_0 = self._get_moved_bounding_box(mb_0, 90 * q.deg)
        transformed_1 = self._get_moved_bounding_box(mb_1, -90 * q.deg)

        def get_concatenated(t_0, t_1, index):
            return np.concatenate((zip(*t_0)[index], zip(*t_1)[index])) * \
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
        n = 100
        t = np.linspace(0, 2 * np.pi, n)

        x_points_0 = t
        y_points_0 = np.cos(t)
        z_points_0 = np.zeros(n)
        traj_0 = Trajectory(zip(x_points_0, y_points_0, z_points_0) * q.m, velocity=1 * q.m / q.s)

        x_points_1 = t
        y_points_1 = 1 + np.cos(t)
        z_points_1 = np.zeros(n)
        traj_1 = Trajectory(zip(x_points_1, y_points_1, z_points_1) * q.m, velocity=1 * q.m / q.s)

        mb_0 = MetaBall(traj_0, 1 * q.m)
        mb_1 = MetaBall(traj_1, 1 * q.m)
        composite = CompositeObject(Trajectory([(0, 0, 0)] * q.m), gr_objects=[mb_0, mb_1])

        # We know the maximum distance for cosine in this case, it's corresponding x and y are
        # x = 2Pi, y = 2
        furthest = np.sqrt(4 * np.pi ** 2 + 4) * q.m + mb_1.furthest_point
        self.assertAlmostEqual(furthest, composite.furthest_point)

    def test_save_transformation_matrix(self):
        old = self.composite.transform_matrix
        self.composite.save_transformation_matrices()

        self.composite.move(1 * q.s)
        self.composite.restore_transformation_matrices()

        np.testing.assert_equal(old, self.composite.transform_matrix)

    @slow
    def test_get_displacement(self):
        p = np.linspace(1, 10, 100)
        x = p
        y = p ** 2
        z = np.zeros(len(p))

        traj = Trajectory(zip(x, y, z) * q.m, velocity=1 * q.m / q.s)
        ball = MetaBall(traj, 1 * q.mm)

        check_distances(ball, 100 * q.mm, 3)

    @slow
    def test_get_next_time(self):
        def move_and_get_position(composite, primitive, abs_time):
            composite.clear_transformation()
            composite.move(abs_time)

            return primitive.position

        n = 100
        p = np.linspace(0, np.pi, n)
        x = np.cos(p)
        y = np.sin(p)
        z = np.zeros(n)
        x_m = 1 + p

        y_m = np.zeros(n)
        z_m = np.zeros(n)
        traj_m = Trajectory(zip(x_m, y_m, z_m) * q.m, velocity=1 * q.m / q.s)
        ball = MetaBall(traj_m, 1 * q.mm)

        traj = Trajectory(zip(x, y, z) * q.m, velocity=1 * q.m / q.s)
        comp = CompositeObject(traj, gr_objects=[ball])

        t_0 = 0 * q.s
        distance = 1000 * q.mm
        while True:
            t_1 = comp.get_next_time(t_0, distance)
            if t_1 is None:
                break

            pos_0 = move_and_get_position(comp, ball, t_0)
            pos_1 = move_and_get_position(comp, ball, t_1)
            diff = np.round(np.max(np.abs(pos_1 - pos_0)).rescale(q.mm) / distance)
            # distance represents a pixel, thus we must less than
            self.assertLessEqual(diff, 1)
            t_0 = t_1
