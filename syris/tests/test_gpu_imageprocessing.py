import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import quantities as q
import syris
from syris.gpu import util as gpu_util
from syris import config as cfg
from syris import imageprocessing as ip
import itertools
from syris.tests import SyrisTest, slow


@slow
class TestGPUImageProcessing(SyrisTest):

    def setUp(self):
        syris.init()
        src = gpu_util.get_source(["vcomplex.cl",
                                   "imageprocessing.cl"])
        self.prg = cl.Program(cfg.OPENCL.ctx, src).build()
        self.size = 256
        self.mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                             size=self.size ** 2 * cfg.PRECISION.cl_float)
        self.res = np.empty((self.size, self.size), dtype=cfg.PRECISION.np_float)
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
        self.prg.gauss_2d_f(cfg.OPENCL.queue,
                          (self.size, self.size),
                           None,
                           self.mem,
                           gpu_util.make_vfloat2(sigma[0], sigma[1]),
                           gpu_util.make_vfloat2(self.pixel_size.simplified,
                                                 self.pixel_size.simplified))

        cl.enqueue_copy(cfg.OPENCL.queue, self.res, self.mem)

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
                        astype(cfg.PRECISION.np_float)
                    im = cl_array.to_device(cfg.OPENCL.queue, im)

                    if coeff:
                        offset = shape[0] / 4, shape[1] / 4
                    else:
                        offset = 0, 0
                    out = ip.bin_image(im, summed_shape, region, offset).get()

                    ground_truth = np.ones_like(out) * np.sum(tile)
                    np.testing.assert_almost_equal(out, ground_truth)

        shape = 16, 16
        summed_shape = 2, 2
        region = 8, 8
        im = np.ones(shape, dtype=cfg.PRECISION.np_float)
        im = cl_array.to_device(cfg.OPENCL.queue, im)
        out = ip.bin_image(im, summed_shape, region, (0, 0), average=True).get()
        ground_truth = np.ones(summed_shape, dtype=cfg.PRECISION.np_float)
        np.testing.assert_almost_equal(out, ground_truth)
