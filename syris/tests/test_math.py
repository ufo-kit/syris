import numpy as np
import quantities as q
from syris import math as smath
from unittest import TestCase


class TestMath(TestCase):

    def test_match_range(self):
        n = 6
        n_1 = 1000
        x_0 = np.linspace(-n / 2, n / 2, n) * q.m
        y_0 = x_0 ** 2
        x_1 = np.linspace(-n / 2, n / 2, n_1) * q.mm
        y_1 = smath.match_range(x_0, y_0, x_1)
        y_1_def = x_1 ** 2

        self.assertEqual(y_0.units, y_1.units)
        self.assertAlmostEqual(np.sum(y_1 - y_1_def), 0 * y_1.units)
