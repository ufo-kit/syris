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

import quantities as q
import numpy as np
from syris.util import make_tuple, get_magnitude, get_gauss
from syris.tests import SyrisTest


class TestUtil(SyrisTest):
    def test_make_tuple(self):
        self.assertEqual(make_tuple(1), (1, 1))
        self.assertEqual(make_tuple(1, num_dims=3), (1, 1, 1))
        self.assertEqual(make_tuple((1, 2)), (1, 2))
        self.assertRaises(ValueError, make_tuple, (1, 1), num_dims=3)

        self.assertEqual(tuple(make_tuple(1 * q.m).simplified.magnitude), (1, 1))
        self.assertEqual(tuple(make_tuple(1 * q.m, num_dims=3).simplified.magnitude), (1, 1, 1))
        self.assertEqual(tuple(make_tuple((1, 2) * q.m).simplified.magnitude), (1, 2))
        self.assertRaises(ValueError, make_tuple, (1, 1) * q.mm, num_dims=3)

    def test_get_magnitude(self):
        self.assertEqual(get_magnitude(1 * q.m), 1)
        self.assertEqual(get_magnitude(1 * q.mm), 0.001)
        self.assertEqual(get_magnitude(1), 1)
        self.assertEqual(tuple(get_magnitude((1, 2) * q.mm)), (0.001, 0.002))

    def test_gauss(self):
        n = 64
        sigma = 2
        mean = n // 2
        x = np.arange(n)
        gt = np.exp(-((x - float(mean)) ** 2) / (2 * sigma ** 2))

        g = get_gauss(x, mean, sigma, normalized=False)
        g_norm = get_gauss(n, mean, sigma, normalized=True)

        np.testing.assert_almost_equal(gt, g)
        self.assertLess(np.abs(np.sum(g_norm) - 1), 1e-7)

        # Extremely broad peak, sum must be 1 anyway
        g = get_gauss(x, mean, n, normalized=True)
        self.assertLess(np.abs(np.sum(g_norm) - 1), 1e-7)
