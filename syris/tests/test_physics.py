import numpy as np
import quantities as q
from syris import physics, config as cfg
from syris.bodies.simple import StaticBody
from syris.devices.sources import BendingMagnet
from syris.geometry import Trajectory
from syris.materials import Material
from syris.tests import default_syris_init, SyrisTest


class TestPhysics(SyrisTest):
    def setUp(self):
        default_syris_init(double_precision=True)
        self.energy = 20 * q.keV
        self.lam = 6.19920937165e-11 * q.m
        self.size = 64
        self.distance = 1 * q.m
        self.pixel_size = 1 * q.um

    def test_energy_to_wavelength(self):
        self.assertAlmostEqual(
            physics.energy_to_wavelength(self.energy).rescale(self.lam.units), self.lam
        )

    def test_wavelength_to_energy(self):
        self.assertAlmostEqual(
            physics.wavelength_to_energy(self.lam).rescale(self.energy.units), self.energy, places=5
        )

    def test_attenuation(self):
        ref_index = 1e-7 + 1e-10j
        energy = 20 * q.keV
        lam = physics.energy_to_wavelength(energy)
        gt = 4 * np.pi * ref_index.imag / lam.simplified
        self.assertAlmostEqual(gt, physics.ref_index_to_attenuation_coeff(ref_index, lam))

    def _get_propagator(self, apply_phase_factor=False, fresnel=True):
        return physics.compute_propagator(
            self.size,
            self.distance,
            self.lam,
            self.pixel_size,
            fresnel=fresnel,
            apply_phase_factor=apply_phase_factor,
            mollified=False,
        ).get()

    def _cpu_propagator(self, phase_factor=1, fresnel=True):
        j, i = np.mgrid[-0.5 : 0.5 : 1.0 / self.size, -0.5 : 0.5 : 1.0 / self.size].astype(
            cfg.PRECISION.np_float
        )

        if fresnel:
            return cfg.PRECISION.np_cplx(phase_factor) * np.fft.fftshift(
                np.exp(
                    -np.pi
                    * self.lam.simplified
                    * self.distance.simplified
                    * (i ** 2 + j ** 2)
                    / self.pixel_size.simplified ** 2
                    * 1j
                )
            )
        else:
            return np.fft.fftshift(
                np.exp(
                    1j
                    * 2
                    * np.pi
                    / self.lam.simplified
                    * self.distance.simplified
                    * np.sqrt(
                        1
                        - (i / self.pixel_size.simplified * self.lam.simplified) ** 2
                        - (j / self.pixel_size.simplified * self.lam.simplified) ** 2
                    )
                )
            )

    def test_no_phase_factor(self):
        res = self._get_propagator()
        cpu = self._cpu_propagator()

        np.testing.assert_almost_equal(res, cpu, 5)

    def test_with_phase_factor(self):
        phase = np.exp(2 * np.pi / self.lam.simplified * self.distance.simplified * 1j)

        res = self._get_propagator(True)
        cpu = self._cpu_propagator(phase)

        np.testing.assert_almost_equal(res, cpu, 5)

    def test_transfer(self):
        thickness = np.linspace(0, 0.01, 16).reshape(4, 4) * q.mm
        energy = 10 * q.keV
        wavelength = physics.energy_to_wavelength(energy)
        refractive_index = 1e-6 + 1e-9j
        wavefield = physics.transfer(thickness, refractive_index, wavelength).get()

        exponent = -2 * np.pi * thickness.simplified / wavelength.simplified
        truth = np.exp(exponent * complex(refractive_index.imag, refractive_index.real))
        np.testing.assert_almost_equal(truth, wavefield)

        # Exponent
        wavefield = physics.transfer(thickness, refractive_index, wavelength, exponent=True).get()
        np.testing.assert_almost_equal(truth, np.exp(wavefield))

    def test_transfer_many(self):
        n = 32
        shape = (n, n)
        ps = 1 * q.um
        energies = np.arange(5, 30) * q.keV
        energy = 10 * q.keV
        lam = physics.energy_to_wavelength(energy)
        # Delta causes phase shift between two adjacent pixels by Pi / 16
        delta = (lam / (32 * ps)).simplified.magnitude
        ri = np.ones_like(energies.magnitude, dtype=complex) * delta + 0j
        material = Material("dummy", ri, energies)
        wedge = np.tile(np.arange(n), [n, 1]) * ps
        wedge = StaticBody(wedge, ps, material=material)

        # Test more objects
        u_many = physics.transfer_many([wedge, wedge], shape, ps, energy).get()
        # 2 objects
        u = wedge.transfer(shape, ps, energy).get() ** 2
        np.testing.assert_almost_equal(u, u_many)

        # Test exponent
        u = physics.transfer_many([wedge], shape, ps, energy, exponent=False).get()
        u_exp = physics.transfer_many([wedge], shape, ps, energy, exponent=True).get()
        np.testing.assert_almost_equal(u, np.exp(u_exp))

    def test_transmission_sampling(self):
        def compute_transmission_function(n, ps, energy, material):
            wedge = np.tile(np.arange(n), [n, 1]) * ps
            wedge = StaticBody(wedge, ps, material=material)

            return wedge.transfer((n, n), ps, energy, exponent=True)

        def compute_distance(n, ps, lam, ps_per_lam):
            ca = (lam / (ps_per_lam * ps)).simplified.magnitude
            alpha = np.arccos(ca)
            theta = np.pi / 2 - alpha
            return (n * ps / (2 * np.tan(theta))).simplified

        n = 32
        ps = 1 * q.um
        energies = np.arange(5, 30) * q.keV
        energy = 10 * q.keV
        lam = physics.energy_to_wavelength(energy)
        # Delta causes phase shift between two adjacent pixels by 2 Pi
        delta = (lam / ps).simplified.magnitude
        ri = np.ones_like(energies.magnitude, dtype=complex) * delta + 0j
        material = Material("dummy", ri, energies)

        # Single object
        u = compute_transmission_function(n, ps, energy, material)
        self.assertFalse(physics.is_wavefield_sampling_ok(u))

        # 4x supersampling => phase shift Pi/2
        u = compute_transmission_function(4 * n, ps / 4, energy, material)
        self.assertTrue(physics.is_wavefield_sampling_ok(u))

        # 4x supersampling with 2 objects => phase shift Pi
        n *= 4
        ps /= 4
        wedge = np.tile(np.arange(n), [n, 1]) * ps
        wedge = StaticBody(wedge, ps, material=material)
        u = physics.transfer_many([wedge, wedge], (n, n), ps, energy, exponent=True)
        self.assertFalse(physics.is_wavefield_sampling_ok(u))

        # X-ray source with a parabolic phase profile
        n = 128
        ps = 1 * q.um
        trajectory = Trajectory([(n / 2, n / 2, 0)] * ps)
        # 1 pixel per wavelength => insufficient sampling
        d = compute_distance(n, ps, lam, 1)

        source = BendingMagnet(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            d,
            1,
            np.array([0.2, 0.8]) * q.mm,
            ps,
            trajectory,
            phase_profile="parabola",
        )
        u = source.transfer((n, n), ps, energy, exponent=True)
        self.assertFalse(physics.is_wavefield_sampling_ok(u))

        # 4 pixel per wavelength => good sampling
        d = compute_distance(n, ps, lam, 4)
        source = BendingMagnet(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            d,
            1,
            np.array([0.2, 0.8]) * q.mm,
            ps,
            trajectory,
            phase_profile="parabola",
        )
        u = source.transfer((n, n), ps, energy, exponent=True)
        self.assertTrue(physics.is_wavefield_sampling_ok(u))

    def test_full_propagator(self):
        gt = self._cpu_propagator(fresnel=False)
        propagator = self._get_propagator(fresnel=False)
        np.testing.assert_almost_equal(gt, propagator, decimal=4)
