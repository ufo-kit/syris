import numpy as np
import quantities as q
import syris
from syris.devices.filters import GaussianFilter, MaterialFilter
from syris.materials import Material
from syris.math import fwnm_to_sigma, sigma_to_fwnm
from syris.physics import energy_to_wavelength
from syris.tests import SyrisTest, slow


@slow
class TestFilters(SyrisTest):

    def setUp(self):
        syris.init(device_index=0)
        self.energies = np.arange(10, 20) * q.keV
        self.energy = 15 * q.keV
        delta = np.linspace(1e-5, 1e-6, len(self.energies))
        beta = np.linspace(1e-8, 1e-9, len(self.energies))
        self.material = Material('foo', delta + beta * 1j, self.energies)
        self.thickness = 1 * q.mm
        self.fltr = MaterialFilter(self.thickness, self.material)

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

    def test_exponent(self):
        transmitted = self.fltr.transfer(None, None, self.energy)
        exponent = self.fltr.transfer(None, None, self.energy, exponent=True)
        self.assertAlmostEqual(transmitted, np.exp(exponent), places=5)

    def test_gaussian(self):
        energies = np.arange(5, 30, 0.1) * q.keV
        energy = 15 * q.keV
        sigma = fwnm_to_sigma(1, n=2) * q.keV
        energy_10 = energy - sigma_to_fwnm(sigma.magnitude, n=10) / 2 * q.keV
        fltr = GaussianFilter(energies, energy, sigma, peak_transmission=0.5)
        u_0 = 10 + 0j

        u_f = fltr.transfer(None, None, energy)
        u = u_0 * u_f
        self.assertAlmostEqual(np.abs(u) ** 2, 50, places=4)
        u_f = fltr.transfer(None, None, energy_10)
        u = u_0 * u_f
        self.assertAlmostEqual(np.abs(u) ** 2, 5, places=4)

        # Test exponent
        u_f = fltr.transfer(None, None, energy, exponent=True)
        u = u_0 * np.exp(u_f)
        self.assertAlmostEqual(np.abs(u) ** 2, 50, places=4)
        u_f = fltr.transfer(None, None, energy_10, exponent=True)
        u = u_0 * np.exp(u_f)
        self.assertAlmostEqual(np.abs(u) ** 2, 5, places=4)
