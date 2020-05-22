import numpy as np
import quantities as q
import unittest
from syris.devices.sources import BendingMagnet, FixedSpectrumSource, Wiggler, XRaySourceError
from syris.geometry import Trajectory
import syris.imageprocessing as ip
from syris.physics import energy_to_wavelength
from syris.tests import are_images_supported, default_syris_init, SyrisTest
from syris.tests.util import get_gauss_2d


def make_phase(n, ps, d, energy, phase_profile):
    y, x = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2] * ps
    x = x.simplified.magnitude
    y = y.simplified.magnitude
    d = d.simplified.magnitude
    lam = energy_to_wavelength(energy).simplified.magnitude
    k = 2 * np.pi / lam

    if phase_profile == "parabola":
        r = (x ** 2 + y ** 2) / (2 * d)
    elif phase_profile == "sphere":
        r = np.sqrt(x ** 2 + y ** 2 + d ** 2)
    else:
        raise ValueError("Unknown phase profile '{}'".format(phase_profile))

    return np.exp(k * r * 1j)


class TestSources(SyrisTest):
    def setUp(self):
        # Double precision needed for spherical phase profile
        default_syris_init(double_precision=True)
        self.dE = 0.1 * q.keV
        self.energies = np.arange(14.8, 15, self.dE.magnitude) * q.keV
        self.trajectory = Trajectory([(0, 0, 0)] * q.m)
        self.ps = 10 * q.um

        self.source = BendingMagnet(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            30 * q.m,
            self.dE,
            np.array([0.2, 0.8]) * q.mm,
            self.ps,
            self.trajectory,
        )

    def test_bending_magnet_approx(self):
        source_2 = BendingMagnet(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            30 * q.m,
            self.dE,
            np.array([0.2, 0.8]) * q.mm,
            self.ps,
            self.trajectory,
            profile_approx=False,
        )

        for e in self.energies:
            u_0 = self.source.transfer((16, 16), self.ps, e, t=0 * q.s).get().real
            u_1 = source_2.transfer((16, 16), self.ps, e, t=0 * q.s).get().real
            perc = u_0 / u_1

            # Allow 0.1 % difference
            np.testing.assert_allclose(perc, 1, rtol=1e-3)

    def test_transfer(self):
        shape = 10
        # No trajectory
        self.source.transfer(shape, self.ps, self.energies[0], t=0 * q.s)
        # Width may be larger
        self.source.transfer((shape, 2 * shape), self.ps, self.energies[0], t=0 * q.s)

        # With trajectory
        n = 16
        x = z = np.zeros(n)
        y = np.linspace(0, 1, n) * q.mm
        tr = Trajectory(
            list(zip(x, y, z)) * q.mm,
            pixel_size=10 * q.um,
            furthest_point=0 * q.m,
            velocity=1 * q.mm / q.s,
        )
        source = BendingMagnet(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            30 * q.m,
            self.dE,
            np.array([0.2, 0.8]) * q.mm,
            self.ps,
            trajectory=tr,
        )
        im_0 = source.transfer(shape, self.ps, self.energies[0], t=0 * q.s).get()
        im_1 = source.transfer(shape, self.ps, self.energies[0], t=tr.time / 2).get()

        # There must be a difference between two different times and given trajectory
        self.assertGreater(np.abs(im_1 - im_0).max(), 0)

        # Test exponent
        u = source.transfer(shape, self.ps, self.energies[0], exponent=False).get()
        u_exp = source.transfer(shape, self.ps, self.energies[0], exponent=True).get()
        np.testing.assert_almost_equal(u, np.exp(u_exp))

    def test_set_phase_profile(self):
        with self.assertRaises(XRaySourceError):
            self.source.phase_profile = "foo"

        self.source.phase_profile = "plane"
        self.source.phase_profile = "parabola"
        self.source.phase_profile = "sphere"

    def test_phase_profile(self):
        n = 64
        shape = (n, n)
        ps = 1 * q.um
        energy = 10 * q.keV
        offset = (n // 2, n // 2) * ps

        def test_one_phase_profile(phase_profile):
            self.source.phase_profile = phase_profile
            phase = np.angle(self.source.transfer(shape, ps, energy, offset=offset).get())
            gt = np.angle(
                make_phase(n, ps, self.source.sample_distance, energy, phase_profile=phase_profile)
            )
            np.testing.assert_almost_equal(phase, gt)

        # Plane wave
        self.source.phase_profile = "plane"
        u = self.source.transfer(shape, ps, energy).get()
        np.testing.assert_almost_equal(np.angle(u), 0)

        test_one_phase_profile("parabola")
        test_one_phase_profile("sphere")

    def test_wiggler(self):
        wiggler = Wiggler(
            2.5 * q.GeV,
            150 * q.mA,
            1.5 * q.T,
            30 * q.m,
            self.dE,
            np.array([0.2, 0.8]) * q.mm,
            self.ps,
            self.trajectory,
            4,
        )
        energy = 10 * q.keV

        w_flux = wiggler.get_flux(energy, 0 * q.mrad, self.ps)
        flux = self.source.get_flux(energy, 0 * q.mrad, self.ps)
        self.assertAlmostEqual((w_flux / flux).magnitude, 4)

        shape = (128, 128)
        w_u = np.abs(wiggler.transfer(shape, self.ps, energy).get()) ** 2
        u = np.abs(self.source.transfer(shape, self.ps, energy).get()) ** 2
        np.testing.assert_almost_equal(w_u / u, 4)


class TestFixedSource(SyrisTest):
    def setUp(self):
        # Double precision needed for spherical phase profile
        default_syris_init()
        self.ps = 1 * q.um
        self.n = 128
        self.size = (100, 100) * q.um
        self.sample_dist = 30 * q.m
        self.dE = 1 * q.keV
        self.energies = np.arange(1, 39, self.dE.magnitude) * q.keV
        self.trajectory = Trajectory([(self.n / 2, self.n / 2, 0)] * self.ps)

    def test_init(self):
        # Wrong number of flux points
        flux = np.arange(len(self.energies) - 1) / q.s
        with self.assertRaises(XRaySourceError):
            FixedSpectrumSource(self.energies, flux, self.sample_dist, self.size, self.trajectory)

        # No pixel size for 3D flux
        flux = np.mgrid[: len(self.energies), : self.n, : self.n] / q.s
        with self.assertRaises(XRaySourceError):
            FixedSpectrumSource(self.energies, flux, self.sample_dist, self.size, self.trajectory)

    def test_get_flux(self):
        flux = np.arange(len(self.energies)) / q.s + 10 / q.s
        source = FixedSpectrumSource(
            self.energies, flux, self.sample_dist, self.size, self.trajectory
        )
        # Out of bounds values
        self.assertAlmostEqual(source.get_flux(self.energies[0] - self.dE, None, None), flux[0])
        self.assertAlmostEqual(source.get_flux(self.energies[-1] + self.dE, None, None), flux[-1])

        # Linear interpolation
        self.assertAlmostEqual(
            source.get_flux(1.2 * q.keV, None, None), 0.8 * flux[0] + 0.2 * flux[1]
        )

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_transfer(self):
        def compare_sampling(gauss, factor):
            shape = (int(factor * self.n),) * 2
            im = ip.compute_intensity(
                source.transfer(shape, self.ps / factor, self.energies[0], check=False)
            ).get()
            self.assertAlmostEqual(im.sum(), gauss.sum(), places=3)
            hd = ip.rescale(gauss, shape).get() / factor ** 2
            np.testing.assert_almost_equal(im, hd, decimal=5)

        gauss = np.fft.fftshift(get_gauss_2d((self.n, self.n), self.n / 10.0))
        flux = np.tile(gauss, [len(self.energies), 1, 1]) / q.s
        source = FixedSpectrumSource(
            self.energies, flux, self.sample_dist, self.size, self.trajectory, pixel_size=self.ps
        )

        # Same sampling
        compare_sampling(gauss, 1)

        # Upsampling
        compare_sampling(gauss, 2)

        # Downsampling
        compare_sampling(gauss, 0.5)
