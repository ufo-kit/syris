import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import quantities as q
import syris
from syris.gpu import util as gpu_util
from syris import config as cfg
from syris import imageprocessing as ip
from syris.util import get_magnitude, make_tuple
import itertools
from syris.tests import SyrisTest, slow


def get_gauss_2d(shape, sigma, pixel_size=None, fourier=False):
    shape = make_tuple(shape)
    sigma = get_magnitude(make_tuple(sigma))
    if pixel_size is None:
        pixel_size = (1, 1)
    else:
        pixel_size = get_magnitude(make_tuple(pixel_size))

    if fourier:
        i = np.fft.fftfreq(shape[1]) / pixel_size[1]
        j = np.fft.fftfreq(shape[0]) / pixel_size[0]
        i, j = np.meshgrid(i, j)

        return np.exp(-2 * np.pi ** 2 * ((i * sigma[1]) ** 2 + (j * sigma[0]) ** 2))
    else:
        x = (np.arange(shape[1]) - shape[1] / 2) * pixel_size[1]
        y = (np.arange(shape[0]) - shape[0] / 2) * pixel_size[0]
        x, y = np.meshgrid(x, y)
        gauss = np.exp(- x ** 2 / (2. * sigma[1] ** 2) - y ** 2 / (2. * sigma[0] ** 2))

        return np.fft.ifftshift(gauss)


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

    def _test_gauss(self, shape, fourier):
        """Test if the gauss in Fourier space calculated on a GPU is
        the same as Fourier transform of a gauss in real space.
        """
        sigma = (shape[0] * self.pixel_size.magnitude,
                 shape[1] / 2 * self.pixel_size.magnitude) * self.pixel_size.units
        if fourier:
            # Make the profile broad
            sigma = (1. / sigma[0].magnitude, 1. / sigma[1].magnitude) * sigma.units
        gauss = ip.get_gauss_2d(shape, sigma, self.pixel_size, fourier=fourier).get()
        gt = get_gauss_2d(shape, sigma, self.pixel_size, fourier=fourier)
        np.testing.assert_almost_equal(gauss, gt)

    def test_gauss(self):
        n = (64, 128, 129)
        for shape in itertools.product(n, n):
            self._test_gauss(shape, False)
            self._test_gauss(shape, True)

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
