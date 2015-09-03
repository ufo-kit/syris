import numpy as np
import pyopencl as cl
import quantities as q
import syris
from syris import physics, config as cfg
from syris.tests import SyrisTest


class TestPhysics(SyrisTest):

    def setUp(self):
        syris.init()
        self.energy = 20 * q.keV
        self.lam = 6.19920937165e-11 * q.m
        self.size = 64
        self.mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                             size=self.size ** 2 * cfg.PRECISION.cl_cplx)
        self.res = np.empty((self.size, self.size), dtype=cfg.PRECISION.np_cplx)
        self.distance = 1 * q.m
        self.pixel_size = 1 * q.um

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
        gt = 4 * np.pi * ref_index.imag / lam.simplified
        self.assertAlmostEqual(gt, physics.ref_index_to_attenuation_coeff(ref_index, lam))

    def _get_propagator(self, apply_phase_factor=False):
        return physics.compute_propagator(self.size, self.distance,
                                          self.lam, self.pixel_size,
                                          apply_phase_factor=apply_phase_factor,
                                          mollified=False).get()

    def _cpu_propagator(self, phase_factor=1):
        j, i = np.mgrid[-0.5:0.5:1.0 / self.size, -0.5:0.5:1.0 / self.size].\
            astype(cfg.PRECISION.np_float)

        return cfg.PRECISION.np_cplx(phase_factor) * \
            np.fft.fftshift(np.exp(- np.pi * self.lam.simplified *
                                   self.distance.simplified *
                                   (i ** 2 + j ** 2) /
                                   self.pixel_size.simplified ** 2 * 1j))

    def test_no_phase_factor(self):
        self.res = self._get_propagator()
        cpu = self._cpu_propagator()

        np.testing.assert_almost_equal(self.res, cpu, 5)

    def test_with_phase_factor(self):
        phase = np.exp(2 * np.pi / self.lam.simplified *
                       self.distance.simplified * 1j)

        self.res = self._get_propagator(True)
        cpu = self._cpu_propagator(phase)

        np.testing.assert_almost_equal(self.res, cpu, 5)

    def test_transfer(self):
        thickness = np.linspace(0, 0.01, 16).reshape(4, 4) * q.mm
        energy = 10 * q.keV
        wavelength = physics.energy_to_wavelength(energy)
        refractive_index = 1e-6 + 1e-9j
        wavefield = physics.transfer(thickness, refractive_index, wavelength).get()

        exponent = - 2 * np.pi * thickness.simplified / wavelength.simplified
        truth = np.exp(exponent * np.complex(refractive_index.imag, refractive_index.real))
        np.testing.assert_almost_equal(truth, wavefield)
