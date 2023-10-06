# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import quantities as q
from syris import math as smath
from syris.tests import SyrisTest
from scipy import interpolate as interp
import math


class TestMath(SyrisTest):
    def test_match_range(self):
        n = 6
        n_1 = 1000
        x_0 = np.linspace(-n // 2, n // 2, n) * q.m
        y_0 = x_0 ** 2
        x_1 = np.linspace(-n // 2, n // 2, n_1) * q.mm
        y_1 = smath.match_range(x_0, y_0, x_1)
        y_1_def = x_1 ** 2

        self.assertEqual(y_0.units, y_1.units)
        self.assertAlmostEqual(np.sum(y_1 - y_1_def), 0 * y_1.units)

    def test_closest(self):
        values = np.array([1, 2, 3, 4])
        self.assertEqual(smath.closest(values, 2), 3)
        self.assertEqual(smath.closest(values, -10), 1)
        self.assertEqual(smath.closest(values, 10), None)

    def test_difference_root(self):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)

        tck = interp.splrep(x, y)

        y_d = 0.2
        places = 5

        # f ascending
        x_0 = np.pi / 4
        smath.difference_root(x_0, tck, y_d)
        self.assertAlmostEqual(
            smath.difference_root(x_0, tck, y_d), math.asin(y_d + math.sin(x_0)), places=places
        )

        # f descending
        x_0 = 3 * np.pi / 4
        self.assertAlmostEqual(
            smath.difference_root(x_0, tck, y_d),
            np.pi - math.asin(math.sin(x_0) - y_d),
            places=places,
        )

    def test_supremum(self):
        # Normal flow
        data = np.array([1, 0, 3, 2])
        self.assertEqual(smath.supremum(1, data), 2)

        # Doesn't exist
        self.assertEqual(smath.supremum(3, data), None)

        # Empty data
        self.assertEqual(smath.supremum(1, np.array([])), None)

    def test_infimum(self):
        # Normal flow
        data = np.array([1, 0, 3, 2])
        self.assertEqual(smath.infimum(2, data), 1)

        # Doesn't exist
        self.assertEqual(smath.infimum(0, data), None)

        # Empty data
        self.assertEqual(smath.infimum(1, np.array([])), None)
