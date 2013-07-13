import numpy as np
import pyopencl as cl
import quantities as q
import syris
from syris.gpu import util as gpu_util
from syris import config as cfg
from syris import imageprocessing as ip
from unittest import TestCase
import itertools


class TestGPUImageProcessing(TestCase):

    def setUp(self):
        syris.init()
        src = gpu_util.get_source(["vcomplex.cl",
                                   "imageprocessing.cl"])
        self.prg = cl.Program(cfg.CTX, src).build()
        self.size = 256
        self.mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                             size=self.size ** 2 * cfg.CL_CPLX)
        self.res = np.empty((self.size, self.size), dtype=cfg.NP_CPLX)
        self.distance = 1 * q.m
        self.lam = 4.9594e-11 * q.m
        self.pixel_size = 1 * q.um

    def _gauss_2d(self, sigma):
        y, x = np.mgrid[-self.size / 2:self.size / 2,
                        -self.size / 2:self.size / 2].astype(np.float32) * \
            self.pixel_size.simplified

        return np.exp(- x ** 2 / (2 * sigma[1] ** 2) -
                      y ** 2 / (2 * sigma[0] ** 2)) /\
                     (2 * np.pi * sigma[0] * sigma[1] /
                      self.pixel_size.simplified ** 2)

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

    def test_sum(self):
        widths = [8, 16, 32]
        region_widths = [1, 2, 4, widths[-1]]

        shapes = list(itertools.product(widths, widths))
        regions = list(itertools.product(region_widths, region_widths))

        for shape in shapes:
            for region in regions:
                for coeff in [0, 1]:
                    summed_shape = [shape[i] / (coeff + 1) / region[i]
                                    for i in range(len(shape))]
                    if summed_shape[0] == 0 or summed_shape[1] == 0:
                        continue

                    # Create such tile, that summing along x and y is
                    # not equal to summing along y and then x.
                    tile = np.arange(region[0] * region[1]).reshape(region)
                    im = np.tile(tile, (shape[0] / region[0],
                                        shape[1] / region[1])).\
                        astype(cfg.NP_FLOAT)
                    mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE |
                                    cl.mem_flags.COPY_HOST_PTR, hostbuf=im)

                    if coeff:
                        offset = shape[0] / 4, shape[1] / 4
                    else:
                        offset = 0, 0
                    out_mem = ip.sum(shape, summed_shape, mem, region, offset)
                    res = np.empty(summed_shape, dtype=cfg.NP_FLOAT)
                    cl.enqueue_copy(cfg.QUEUE, res, out_mem)
                    mem.release()
                    out_mem.release()

                    ground_truth = np.ones_like(res) * np.sum(tile)
                    np.testing.assert_almost_equal(res, ground_truth)

        shape = 16, 16
        summed_shape = 2, 2
        region = 8, 8
        im = np.ones(shape, dtype=cfg.NP_FLOAT)
        mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE |
                        cl.mem_flags.COPY_HOST_PTR, hostbuf=im)
        out_mem = ip.sum(shape, summed_shape, mem, region,
                         (0, 0), average=True)
        res = np.empty(summed_shape, dtype=cfg.NP_FLOAT)
        cl.enqueue_copy(cfg.QUEUE, res, out_mem)
        ground_truth = np.ones(summed_shape, dtype=cfg.NP_FLOAT)
        np.testing.assert_almost_equal(res, ground_truth)
