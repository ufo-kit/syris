import numpy as np
import quantities as q
import syris
from syris.opticalelements.samples import Sample
from syris.opticalelements.geometry import Trajectory
from syris.opticalelements.graphicalobjects import MetaBall, CompositeObject
from syris.tests.base import SyrisTest


class TestSamples(SyrisTest):

    def setUp(self):
        syris.init()
        self.shape = 2, 2
        self.pixel_size = 1e-3 * q.mm
        self.moving = Sample({}, self.shape, self.pixel_size)

    def test_moving_sample_parts(self):
        self.assertEqual(self.moving.materials, [])
        self.moving.add("PMMA", 1)
        self.moving.add("PMMA", 2)
        self.moving.add("glass", 3)

        # Immutability test.
        self.assertEqual(self.moving.get_objects("PMMA").__class__, tuple)

        # Adding objects.
        self.assertEqual(self.moving.get_objects("PMMA"), (1, 2))
        self.moving.add("PMMA", 4)
        self.assertEqual(self.moving.get_objects("PMMA"), (1, 2, 4))

        # Adding new materials.
        self.moving.add("new", 5)
        self.assertEqual(self.moving.get_objects("new"), (5,))

        # Objects.
        np.testing.assert_equal(np.sort(self.moving.objects), np.arange(1, 6))

    def test_moved_materials(self):
        def get_linear_trajectory(velocity):
            x = np.linspace(0, 1, 10)
            y = z = np.zeros(len(x))

            c_points = zip(x, y, z) * q.mm

            return Trajectory(c_points, velocity=velocity)

        t_0 = get_linear_trajectory(1e1 * q.mm / q.s)
        t_1 = get_linear_trajectory(1 * q.mm / q.s)
        t_2 = get_linear_trajectory(1e-1 * q.mm / q.s)
        t_stat = Trajectory([(0, 0, 0)] * q.mm, 0 * q.mm)

        mb_0 = MetaBall(t_0, 1.0 * q.mm)
        mb_1 = MetaBall(t_1, 1.0 * q.mm)
        mb_2 = MetaBall(t_2, 1.0 * q.mm)
        mb_stat = MetaBall(t_stat, 1 * q.mm)

        mat_0 = "PMMA"
        mat_1 = "glass"
        mat_2 = "PVC"
        mat_stat = "stat"

        self.moving.add(mat_0, mb_0)
        self.moving.add(mat_1, mb_1)
        self.moving.add(mat_2, mb_2)
        self.moving.add(mat_stat, mb_stat)

        ultra_fast = self.moving.get_moved_materials(0 * q.s, 1e-5 * q.s)
        fast = self.moving.get_moved_materials(0 * q.s, 1e-4 * q.s)
        normal = self.moving.get_moved_materials(0 * q.s, 1e-3 * q.s)
        slow = self.moving.get_moved_materials(0 * q.s, 1 * q.s)

        self.assertEqual(set([]), set(ultra_fast))
        self.assertEqual(set([mat_0]), set(fast))
        self.assertEqual(set([mat_0, mat_1]), set(normal))
        self.assertEqual(set([mat_0, mat_1, mat_2]), set(slow))

        # Also test composite, make it of the slow objects, thus
        # it starts moving later.
        comp = CompositeObject(t_stat, gr_objects=[mb_1, mb_2])
        sample = Sample({mat_0: comp}, self.shape, self.pixel_size)
        ultra_fast = sample.get_moved_materials(0 * q.s, 1e-5 * q.s)
        fast = sample.get_moved_materials(0 * q.s, 1e-4 * q.s)
        slow = sample.get_moved_materials(0 * q.s, 1 * q.s)
        self.assertEqual(set([]), set(ultra_fast))
        self.assertEqual(set([]), set(fast))
        self.assertEqual(set([mat_0]), set(slow))
