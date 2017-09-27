import numpy as np
import quantities as q
import syris
from syris.bodies.simple import StaticBody
from syris.materials import Material
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength, transfer
from syris.tests import SyrisTest, opencl, slow


class DummyOpticalElement(OpticalElement):

    def _transfer(self, shape, pixel_size, energy, offset, exponent=False, t=None, queue=None,
                  out=None, check=True, block=False):
        return shape, pixel_size


@slow
class TestOpticalElement(SyrisTest):

    def setUp(self):
        syris.init(device_index=0)
        energies = range(10, 20) * q.keV
        self.energy = energies[len(energies) / 2]
        self.material = Material('foo', np.arange(len(energies), dtype=np.complex), energies)

    def test_2d_conversion(self):
        elem = DummyOpticalElement()
        shape, ps = elem.transfer(1, 1 * q.m, 0 * q.keV)
        self.assertEqual(len(shape), 2)
        self.assertEqual(len(ps), 2)

    @opencl
    def test_transfer(self):
        go = StaticBody(np.arange(4 ** 2).reshape(4, 4) * q.um, 1 * q.um, material=self.material)
        transferred = go.transfer((4, 4), 1 * q.um, self.energy).get()
        gt = transfer(go.thickness, self.material.get_refractive_index(self.energy),
                      energy_to_wavelength(self.energy)).get()
        np.testing.assert_almost_equal(gt, transferred)
