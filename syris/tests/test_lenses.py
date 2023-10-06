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
from syris.devices.lenses import Lens
from syris.tests import SyrisTest


class TestLenses(SyrisTest):
    def setUp(self):
        self.f_length = 100 * q.mm
        self.f_number = 1.4
        self.magnification = 1
        self.transmission_eff = 0.5
        self.psf_sigma = (1, 1) * q.um
        self.lens = Lens(
            self.magnification,
            f_number=self.f_number,
            focal_length=self.f_length,
            transmission_eff=self.transmission_eff,
            sigma=self.psf_sigma,
        )

    def test_init(self):
        self.assertRaises(
            ValueError, Lens, self.f_number, self.f_length, self.magnification, -1, self.psf_sigma
        )
        self.assertRaises(
            ValueError, Lens, self.f_number, self.f_length, self.magnification, 10, self.psf_sigma
        )

    def test_numerical_aperture(self):
        self.assertAlmostEqual(self.lens.numerical_aperture, 0.17579063848)
        self.lens.magnification = 3
        self.assertAlmostEqual(self.lens.numerical_aperture, 0.25873608522)

    def test_given_numerical_aperture(self):
        gt = 1
        lens = Lens(1, na=gt)
        self.assertEqual(gt, lens.numerical_aperture)

    def test_constructor_args(self):
        self.assertRaises(ValueError, Lens, 1)
        Lens(1, na=1)
        Lens(1, f_number=1, focal_length=1 * q.m)

    def test_na_supersedes_compute(self):
        gt = 0.5
        lens = Lens(1, na=gt, f_number=2, focal_length=100 * q.mm)
        self.assertEqual(gt, lens.na)
