import numpy as np
from numpy import linalg
import quantities as q
from syris.opticalelements import geometry as geom
from syris.opticalelements.geometry import Trajectory
from syris.opticalelements.graphicalobjects import MetaBall, CompositeObject
from unittest import TestCase
import itertools


def get_control_points():
    return np.array([(0, 0, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (0, 0, 1),
                     (1, 1, 1)]) * q.mm


def get_linear(direction, start=(0, 0, 0), num=4):
    res = []
    for i in range(num):
        point = np.copy(start)
        point[direction] += i
        res.append(point)

    return np.array(res) * q.mm


class TestGraphicalObjects(TestCase):

    def setUp(self):
        self.pixel_size = 1 * q.um

        control_points = get_linear(geom.X, start=(1, 1, 1))

        traj = Trajectory(control_points, velocity=1 * q.mm / q.s)
        self.metaball = MetaBall(traj, 1 * q.mm)

        self.metaball_2 = MetaBall(Trajectory(get_linear(geom.Z)), 2 * q.mm)

        self.composite = CompositeObject(traj,
                                         gr_objects=[self.metaball,
                                                     self.metaball_2])

    def test_metaball(self):
        self.assertEqual(self.metaball.radius, 1 * q.mm)
        np.testing.assert_almost_equal(self.metaball.center,
                                       np.array([1, 1, 1]) * q.mm)

    def test_metaball_bounding_box(self):
        """Bounding box moves along with its object."""
        mb = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 0.5 * q.mm)
        transformed = self._get_moved_bounding_box(mb, 90 * q.deg)

        np.testing.assert_almost_equal(transformed, mb.bounding_box.points)

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

    def test_composite_subobjects(self):
        traj = Trajectory(get_linear(geom.X))
        composite = CompositeObject(traj,
                                    gr_objects=[self.composite, self.metaball])

        primitive = composite.primitive_objects
        self.assertTrue(self.metaball in primitive)
        self.assertTrue(self.metaball_2 in primitive)
        self.assertEqual(len(composite.primitive_objects), 2)

    def test_composite_add(self):
        def add():
            self.composite.primitive_objects[0] = self.metaball_2
        self.assertRaises(TypeError, add)
        composite_2 = CompositeObject(self.composite.trajectory)
        composite_2.add(self.composite)
        self.assertRaises(ValueError, self.composite.add, composite_2)
        composite_2.add(composite_2)
        self.assertEqual(len(composite_2.objects), 1)

    def test_move(self):
        mb = MetaBall(Trajectory(get_linear(geom.Y), velocity=1 * q.mm / q.s),
                      1 * q.mm)
        self.assertAlmostEqual(mb.get_next_time(0 * q.s, self.pixel_size),
                               1e-3 * q.s)

    def test_trajectory_less_than_pixel(self):
        points = get_linear(geom.Y) / 1e6
        mb = MetaBall(Trajectory(points, velocity=1 * q.mm / q.s),
                      1 * q.mm)
        self.assertEqual(mb.get_next_time(0 * q.s, self.pixel_size), None)

    def test_composite_move(self):
        # Composite object movement must affect its subobjects.
        abs_time = 0 * q.s
        self.composite.move(abs_time)
        np.testing.assert_almost_equal(self.composite.position,
                                       np.array([1, 1, 1]) * q.mm)
        np.testing.assert_almost_equal(self.metaball.position,
                                       np.array([2, 0, 2]) * q.mm)

        self.composite.clear_transformation()

        abs_time = self.composite.trajectory.time
        self.composite.move(abs_time)
        np.testing.assert_almost_equal(self.composite.position,
                                       np.array([4, 1, 1]) * q.mm)
        np.testing.assert_almost_equal(self.metaball.position,
                                       np.array([5, -3, 2]) * q.mm)

    def test_move_limits(self):
        self.metaball.clear_transformation()
        self.metaball.move(self.metaball.trajectory.time)
        pos_0 = self.metaball.position

        self.metaball.clear_transformation()
        # Beyond the trajectory end.
        self.metaball.move(2 * self.metaball.trajectory.time)
        pos_1 = self.metaball.position

        np.testing.assert_almost_equal(pos_0, pos_1)

    def test_moved_only_rotation(self):
        """Test for movement in a setting where an object stays at
        one position but its "up" vector moved (object only rotates).
        """
        base = np.linspace(0, np.pi, 10)
        x = np.cos(base)
        y = np.sin(base)
        z = np.zeros(len(base))

        c_points = zip(x, y, z) * q.mm

        traj = Trajectory(c_points, velocity=1 * q.mm / q.s)
        metaball = MetaBall(Trajectory([-traj.control_points[0].magnitude] *
                                       traj.control_points.units), 1 * q.mm)

        comp = CompositeObject(traj, gr_objects=[metaball])
        self.assertTrue(comp.moved(0 * q.s, traj.time / 2, 100 * q.um))

    def _get_moved_bounding_box(self, obj, angle):
        obj.scale((0.75, 0.5, 1.2))
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

    def test_movement_negation(self):
        s_pos = get_linear(geom.Y)
        s_neg = -s_pos

        t_pos = Trajectory(s_pos, velocity=1 * q.mm / q.s)
        t_neg = Trajectory(s_neg, velocity=1 * q.mm / q.s)

        m_neg = MetaBall(t_neg, 1 * q.mm)
        co = CompositeObject(t_pos, gr_objects=[m_neg])

        self.assertEqual(co.get_next_time(0 * q.s, self.pixel_size), None)

    def test_nonlinear_movement(self):
        """
        Translational movement combined with the movement coming from
        rotation.
        """
        t = np.linspace(-np.pi / 2, np.pi / 2, 10)
        x = np.sin(t)
        y = np.cos(t)

        c_points = zip(x, y, len(x) * [0]) * q.mm

        mb = MetaBall(Trajectory(c_points, velocity=1 * q.mm / q.s), 1 * q.mm)
        self.assertNotEqual(mb.get_next_time(0 * q.s, self.pixel_size),
                            mb.trajectory.get_next_time(0 * q.s,
                                                        self.pixel_size))
