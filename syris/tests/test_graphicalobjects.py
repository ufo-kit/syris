import numpy as np
import quantities as q
from syris.opticalelements import geometry as geom
from syris.opticalelements.geometry import Trajectory
from syris.opticalelements.graphicalobjects import MetaBall, CompositeObject
from syris.tests.base import SyrisTest
import itertools
from numpy import linalg
from syris.tests.graphics_util import get_linear_points


def get_control_points():
    return np.array([(0, 0, 0),
                     (1, 0, 0),
                     (1, 1, 0),
                     (0, 0, 1),
                     (1, 1, 1)]) * q.mm


class TestGraphicalObjects(SyrisTest):

    def setUp(self):
        self.pixel_size = 1 * q.um

        control_points = get_linear_points(geom.X, start=(1, 1, 1))

        traj = Trajectory(control_points, velocity=1 * q.mm / q.s)
        self.metaball = MetaBall(traj, 1 * q.mm)

        self.metaball_2 = MetaBall(Trajectory(get_linear_points(geom.Z)), 2 * q.mm)

        self.composite = CompositeObject(traj,
                                         gr_objects=[self.metaball,
                                                     self.metaball_2])
        
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
 
        c_1 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_1, m_2])
        c_2 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_3])
        c_3 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[c_1, c_2])
        c_4 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_4, m_5])
        c_5 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[c_3, c_4])
 
        # empty object
        CompositeObject(Trajectory([(0, 0, 0)] * q.mm))
 
        # non-uniform objects list in the constructor
        with self.assertRaises(TypeError) as ctx:
            CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                            gr_objects=[m_1, c_1])
        self.assertEqual("Composite object direct children " +
                         "must be all of the same type",
                         ctx.exception.message)
 
        # Last composite objects which have only primitive children.
        g_t = [c_1, c_2, c_4]
        self.assertEqual(set(g_t), set(c_5.get_last_composites()))
 
        # Only one composite object and one child.
        self.assertEqual(c_2, c_2.get_last_composites()[0])
 
        # Empty composite object.
        c_empty = CompositeObject(Trajectory([(0, 0, 0)] * q.mm))
        self.assertEqual([], c_empty.get_last_composites())
 
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
 
        # Add primitive to composite node.
        m_6 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 5 * q.mm)
        with self.assertRaises(TypeError) as ctx:
            c_5.add(m_6)
        self.assertEqual("Composite object direct children " +
                         "must be all of the same type",
                         ctx.exception.message)
 
        # Add composite to primitive node.
        c_6 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_6])
        with self.assertRaises(TypeError) as ctx:
            c_2.add(c_6)
        self.assertEqual("Composite object direct children " +
                         "must be all of the same type",
                         ctx.exception.message)
        
    def test_parents(self):
        m_1 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 1 * q.mm)
        m_2 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 2 * q.mm)
        m_3 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 3 * q.mm)
        
        c_1 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_1])
        c_2 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[m_2, m_3])
        root = CompositeObject(Trajectory([(0, 0, 0)] * q.mm),
                              gr_objects=[c_1, c_2])
        
        # parent links
        self.assertEqual(m_3.parent, c_2)
        self.assertEqual(m_3.parent.parent, root)
        
        # root parent finding
        self.assertEqual(m_1.root, root)
        self.assertEqual(m_3.root, root)
        
        # no root
        m_4 = MetaBall(Trajectory([(0, 0, 0)] * q.mm), 4 * q.mm)
        self.assertEqual(m_4.root, m_4)
        c_3 = CompositeObject(Trajectory([(0, 0, 0)] * q.mm))
        self.assertEqual(c_3.root, c_3)
        
 
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
  
    def test_move(self):
        mb = MetaBall(Trajectory(get_linear_points(geom.Y),
                                 velocity=1 * q.mm / q.s), 1 * q.mm)
        self.assertAlmostEqual(mb.get_next_time(0 * q.s, self.pixel_size),
                               1e-3 * q.s)
  
    def test_trajectory_less_than_pixel(self):
        points = get_linear_points(geom.Y) / 1e6
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
        """
        Test for movement in a setting where an object stays at
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
     
    def test_composite_next_time(self):
        self.assertAlmostEqual(self.composite.get_next_time(0 * q.s,
                                              self.pixel_size), 1e-3 * q.s)
 
    def test_movement_negation(self):
        s_pos = get_linear_points(geom.Y)
        s_neg = -s_pos
  
        t_pos = Trajectory(s_pos, velocity=1 * q.mm / q.s)
        t_neg = Trajectory(s_neg, velocity=1 * q.mm / q.s)
  
        m_neg = MetaBall(t_neg, 1 * q.mm)
        co = CompositeObject(t_pos, gr_objects=[m_neg])
  
        self.assertEqual(co.get_next_time(0 * q.s, self.pixel_size), None)
  
    def test_complex_movement(self):
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
