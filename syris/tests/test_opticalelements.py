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
import pyopencl.array as cl_array
import quantities as q
import syris.config as cfg
from syris.bodies.simple import StaticBody
from syris.materials import Material
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength, transfer
from syris.tests import default_syris_init, SyrisTest


class DummyOpticalElement(OpticalElement):
    def _transfer(
        self,
        shape,
        pixel_size,
        energy,
        offset,
        exponent=False,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        return shape, pixel_size

    def _transfer_fourier(
        self, shape, pixel_size, energy, t=None, queue=None, out=None, block=False
    ):
        if out is None:
            out = cl_array.zeros(queue, shape, cfg.PRECISION.np_cplx)

        return out


class TestOpticalElement(SyrisTest):
    def setUp(self):
        default_syris_init()
        energies = list(range(10, 20)) * q.keV
        self.energy = energies[len(energies) // 2]
        self.material = Material("foo", np.arange(len(energies), dtype=complex), energies)

    def test_2d_conversion(self):
        elem = DummyOpticalElement()
        shape, ps = elem.transfer(1, 1 * q.m, 0 * q.keV)
        self.assertEqual(len(shape), 2)
        self.assertEqual(len(ps), 2)

    def test_transfer(self):
        go = StaticBody(np.arange(4 ** 2).reshape(4, 4) * q.um, 1 * q.um, material=self.material)
        transferred = go.transfer((4, 4), 1 * q.um, self.energy).get()
        gt = transfer(
            go.thickness,
            self.material.get_refractive_index(self.energy),
            energy_to_wavelength(self.energy),
        ).get()
        np.testing.assert_almost_equal(gt, transferred)

    def test_transfer_fourier(self):
        elem = DummyOpticalElement()
        print(elem.__class__._transfer_fourier == OpticalElement._transfer_fourier)
        u = elem.transfer_fourier((4, 4), 1 * q.um, 10 * q.keV).get()
        np.testing.assert_almost_equal(u, 0 + 0j)
