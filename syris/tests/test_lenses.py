import quantities as q
from syris.devices.lenses import Lens
from unittest import TestCase


class TestLenses(TestCase):

    def test_init(self):
        self.assertRaises(ValueError, Lens, -10, None, 1, None)
        self.assertRaises(ValueError, Lens, 10, None, 1, None)
        self.assertRaises(ValueError, Lens, 0.5, 0.5, 1, None)
        Lens(0.5, 1.5, 1, (1 * q.m, 1 * q.m))
