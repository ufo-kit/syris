import numpy as np
import pyopencl as cl
import quantities as q
import syris
from syris.gpu import util as gpu_util
from syris import physics, config as cfg
from unittest import TestCase


class TestGPUImageProcessing(TestCase):

    def setUp(self):
        syris.init()
        src = gpu_util.get_source(["vcomplex.cl",
                                   "imageprocessing.cl",
                                   "physics.cl"])
        self.prg = cl.Program(cfg.CTX, src).build()
        self.size = 256
        self.mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                             size=self.size ** 2 * cfg.CL_CPLX)
        self.res = np.empty((self.size, self.size), dtype=cfg.NP_CPLX)
        self.distance = 1 * q.m
        self.lam = 4.9594e-11 * q.m
        self.pixel_size = 1 * q.um

    def _get_propagator(self, apply_phase_factor=False):
        return physics.get_propagator(self.size, self.distance,
                                      self.lam, self.pixel_size,
                                      apply_phase_factor,
                                      copy_to_host=True)

    def _cpu_propagator(self, phase_factor=1):
        j, i = np.mgrid[-0.5:0.5:1.0 / self.size, -0.5:0.5:1.0 / self.size].\
            astype(cfg.NP_FLOAT)

        return cfg.NP_CPLX(phase_factor) * \
            np.fft.fftshift(np.exp(- np.pi * self.lam.simplified *
                                   self.distance.simplified *
                                   (i ** 2 + j ** 2) /
                                   self.pixel_size.simplified ** 2 * 1j))

    def _gauss_2d(self, sigma):
        y, x = np.mgrid[-self.size / 2:self.size / 2,
                        -self.size / 2:self.size / 2].astype(np.float32) * \
            self.pixel_size.simplified

        return np.exp(- x ** 2 / (2 * sigma[1] ** 2) -
                      y ** 2 / (2 * sigma[0] ** 2)) /\
                     (2 * np.pi * sigma[0] * sigma[1] /
                      self.pixel_size.simplified ** 2)

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

    def test_gauss(self):
        """Test if the gauss in Fourier space calculated on a GPU is
        the same as Fourier transform of a gauss in real space.
        """
        sigma = 10 * self.pixel_size.simplified, 5 * self.pixel_size.simplified
        self.prg.gauss_2_f(cfg.QUEUE,
                          (self.size, self.size),
                           None,
                           self.mem,
                           gpu_util.make_vfloat2(sigma[0], sigma[1]),
                           cfg.NP_FLOAT(self.pixel_size.simplified))

        cl.enqueue_copy(cfg.QUEUE, self.res, self.mem)

        cpu = np.fft.fftshift(self._gauss_2d((sigma[1], sigma[0])))
        cpu_f = np.fft.fft2(cpu)

        np.testing.assert_almost_equal(self.res, cpu_f)

        # Normalization test. The sum of the Gaussian must be 1.
        gauss_real = np.fft.ifft2(self.res)
        self.assertAlmostEqual(np.sum(gauss_real), 1)
