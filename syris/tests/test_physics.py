import quantities as q
from syris import physics
from unittest import TestCase


class TestPhysics(TestCase):

    def setUp(self):
        self.energy = 20 * q.keV
        self.lam = 6.19920937165e-11 * q.m

    def test_energy_to_wavelength(self):
        self.assertAlmostEqual(physics.energy_to_wavelength(self.energy).
                               rescale(self.lam.units), self.lam)

    def test_wavelength_to_energy(self):
        self.assertAlmostEqual(physics.wavelength_to_energy(self.lam).
                               rescale(self.energy.units), self.energy,
                               places=5)

    def test_attenuation(self):
        ref_index = 1e-7 + 1e-10j
        energy = 20 * q.keV
        lam = physics.energy_to_wavelength(energy)
        self.assertAlmostEqual(physics.
                               ref_index_to_attenuation_coeff(
                                   ref_index, energy),
                               physics.ref_index_to_attenuation_coeff(
                               ref_index, lam))
