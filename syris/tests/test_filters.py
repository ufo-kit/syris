import numpy as np
import quantities as q
import syris
from syris.devices.filters import Filter
from syris.materials import Material
from syris.physics import energy_to_wavelength
from syris.tests import SyrisTest, slow


@slow
class TestFilters(SyrisTest):

    def setUp(self):
        syris.init(device_index=0)
        self.energies = np.arange(10, 20) * q.keV
        self.energy = 15 * q.keV
        delta = np.random.uniform(low=1e-5, high=1e-6, size=len(self.energies))
        beta = np.random.uniform(low=1e-8, high=1e-9, size=len(self.energies))
        self.material = Material('foo', delta + beta * 1j, self.energies)
        self.thickness = 1 * q.mm
        self.fltr = Filter(self.thickness, self.material)

    def test_transfer(self):
        thickness = self.thickness.simplified.magnitude
        lam = energy_to_wavelength(self.energy).simplified.magnitude
        coeff = -2 * np.pi * thickness / lam
        ri = self.material.get_refractive_index(self.energy)
        gt = np.exp(coeff * (ri.imag + ri.real * 1j))
        fltr = self.fltr.transfer(None, None, self.energy)

        self.assertAlmostEqual(gt, fltr)

    def test_get_attenuation(self):
        att = self.material.get_attenuation_coefficient(self.energy).simplified.magnitude
        thickness = self.thickness.simplified.magnitude
        gt = att * thickness
        fltr = self.fltr.get_attenuation(self.energy)

        self.assertAlmostEqual(gt, fltr)
