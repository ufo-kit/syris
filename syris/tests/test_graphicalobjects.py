import numpy as np
import quantities as q
from syris.opticalelements import geometry as geom
from syris.opticalelements.geometry import Trajectory
from syris.opticalelements.graphicalobjects import GraphicalObject, \
    MetaBall, CompositeObject
from unittest import TestCase


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

        points, length = geom.interpolate_points(get_linear(geom.X,
                                                            start=(1, 1, 1)),
                                                 self.pixel_size)

        traj = Trajectory(points, length, [(length, 1 * q.mm / q.s)])
        self.metaball = MetaBall(traj, 1 * q.mm)

        points, length = geom.interpolate_points(get_linear(geom.Z),
                                                 self.pixel_size)
        traj = Trajectory(points, length, [(length, 1 * q.mm / q.s)])
        self.metaball_2 = MetaBall(traj, 2 * q.mm)

        points, length = geom.interpolate_points(get_linear(geom.X,
                                                            start=(1, 1, 1)),
                                                 self.pixel_size)

        traj = Trajectory(points, length, [(length, 1 * q.mm / q.s)])
        self.composite = CompositeObject(traj,
                                         gr_objects=[self.metaball,
                                                     self.metaball_2])

    def test_moved(self):
        c_points = get_control_points()
        points, length = geom.interpolate_points(c_points, self.pixel_size)

        traj = Trajectory(points, length, [(length, 5 * q.mm / q.s)])
        g_obj = GraphicalObject(traj)
        t_0 = 0 * q.s
        for t_1 in np.linspace(0, 0.1, 10) * q.s:
            p_0 = traj.get_point(t_0)
            p_1 = traj.get_point(t_1)
            ground_truth = True if geom.length(p_1 - p_0) > self.pixel_size \
                else False
            self.assertEqual(g_obj.moved(t_0, t_1, self.pixel_size),
                             ground_truth)

    def test_metaball(self):
        self.assertEqual(self.metaball.radius, 1 * q.mm)
        np.testing.assert_almost_equal(self.metaball.center,
                                       np.array([1, 1, 1]) * q.mm)

    def test_composite_subobjects(self):
        points, length = geom.interpolate_points(get_linear(geom.X),
                                                 self.pixel_size)
        traj = Trajectory(points, length, [(length, 1 * q.mm / q.s)])
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

    def test_composite_moved(self):
        pixel_size = 1 * q.um
        times = np.linspace(0, 0.1, 10) * q.s
        previous = {}
        matrices = {}

        for i in range(len(times)):
            self.composite.clear_transformation()
            self.composite.move(times[i])
            for obj in self.composite.objects:
                matrices[obj] = np.copy(obj.transform_matrix)

            ground_truth = False
            if previous != {}:
                for obj in self.composite.objects:
                    if obj.__class__ != CompositeObject:
                        if geom.length(obj.position - previous[obj]) > \
                                pixel_size:
                            ground_truth = True
                            break
            else:
                for obj in self.composite.primitive_objects:
                    previous[obj] = np.copy(obj.position) * obj.position.units

            moved = self.composite.moved(0 * q.s, times[i], pixel_size)

            self.assertEqual(moved, ground_truth)

            # The transformation matrix must be unchanged after movement
            # testing.
            for obj in self.composite.objects:
                np.testing.assert_almost_equal(obj.transform_matrix,
                                               matrices[obj])
