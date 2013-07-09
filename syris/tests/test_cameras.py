import numpy as np
import quantities as q
from syris.devices.cameras import Camera, FPSError, is_fps_feasible
from testfixtures import ShouldRaise
from unittest import TestCase


class TestCamera(TestCase):

    def setUp(self):
        self.camera = Camera((64, 64),
                             1e-3 * q.um,
                             1.0,
                             10.0,
                             1.0,
                             10,
                             None,
                             1 * q.ms,
                             1000 / q.s)

    def test_is_fps_feasible(self):
        self.assertTrue(is_fps_feasible(1000 / q.s, 1 * q.ms))
        self.assertTrue(is_fps_feasible(1000 / q.s, 0.5 * q.ms))
        self.assertFalse(is_fps_feasible(1000 / q.s, 2 * q.ms))

    def test_fps_setting(self):
        self.camera.fps = 500 / q.s

        with ShouldRaise(FPSError):
            self.camera.fps = 1001 / q.s

    def test_exp_time_setting(self):
        self.camera.exp_time = 0.5 * q.ms

        with ShouldRaise(FPSError):
            self.camera.exp_time = 1.5 * q.ms

    def test_bpp(self):
        self.assertEqual(self.camera.max_grey_value, 2 ** self.camera.bpp - 1)

    def test_dark(self):
        photons = np.zeros(self.camera.shape)
        res = self.camera.get_image(photons)

        self.assertNotEqual(np.var(res), 0)
        self.assertNotEqual(np.sum(res), 0)

    def test_saturation(self):
        photons = np.ones(self.camera.shape) * 10 * self.camera.max_grey_value
        res = self.camera.get_image(photons)

        self.assertEqual(np.var(res), 0)

        diff = np.ones(self.camera.shape) * self.camera.max_grey_value - res
        self.assertEqual(np.sum(diff), 0)
