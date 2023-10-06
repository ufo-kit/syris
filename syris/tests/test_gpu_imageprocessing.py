# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pyopencl.array as cl_array
import quantities as q
import unittest
from syris.gpu import util as gpu_util
from syris import config as cfg
from syris import imageprocessing as ip
from syris.math import fwnm_to_sigma
import itertools
from syris.tests import are_images_supported, default_syris_init, SyrisTest
from syris.tests.util import get_gauss_2d


def bin_cpu(image, shape):
    factor = (image.shape[0] // shape[0], image.shape[1] // shape[1])
    im = np.copy(image)
    for k in range(1, factor[0]):
        im[:: factor[0], :] += im[k :: factor[0], :]
    for k in range(1, factor[1]):
        im[:, :: factor[1]] += im[:, k :: factor[1]]
    return im[:: factor[0], :: factor[1]]


def rescale_scipy(image, factor):
    from scipy.interpolate import RectBivariateSpline

    m, n = image.shape
    x = np.arange(n)
    y = np.arange(m)
    hd_y = np.arange(factor[0] * m) / float(factor[0]) - 0.5 + 1.0 / (2.0 * factor[0])
    hd_x = np.arange(factor[1] * n) / float(factor[1]) - 0.5 + 1.0 / (2.0 * factor[1])
    spl = RectBivariateSpline(y, x, image, kx=1, ky=1)

    return spl(hd_y.astype(cfg.PRECISION.np_float), hd_x.astype(cfg.PRECISION.np_float), grid=True)


class TestGPUImageProcessing(SyrisTest):
    def setUp(self):
        default_syris_init()
        self.pixel_size = 1 * q.um

    def _test_gauss(self, shape, fourier):
        """Test if the gauss in Fourier space calculated on a GPU is
        the same as Fourier transform of a gauss in real space.
        """
        sigma = (
            shape[0] * self.pixel_size.magnitude,
            shape[1] / 2 * self.pixel_size.magnitude,
        ) * self.pixel_size.units
        if fourier:
            # Make the profile broad
            sigma = (1.0 / sigma[0].magnitude, 1.0 / sigma[1].magnitude) * sigma.units
        gauss = ip.get_gauss_2d(shape, sigma, self.pixel_size, fourier=fourier).get()
        gt = get_gauss_2d(shape, sigma, self.pixel_size, fourier=fourier)
        np.testing.assert_almost_equal(gauss, gt)

    def test_gauss(self):
        n = (64, 128, 129)
        for shape in itertools.product(n, n):
            self._test_gauss(shape, False)
            self._test_gauss(shape, True)

    def test_sum(self):
        m = 8
        n = 16
        image = np.arange(m * n).reshape(m, n).astype(cfg.PRECISION.np_float)
        cl_im = cl_array.to_device(cfg.OPENCL.queue, image)
        sizes = (1, 2, 4)
        for shape in itertools.product(sizes, sizes):
            region = (m // shape[0], n // shape[1])
            gt = bin_cpu(image, shape)
            res = ip.bin_image(cl_im, shape)
            np.testing.assert_equal(gt, res)

            # Test averaging
            res = ip.bin_image(cl_im, shape, average=True)
            np.testing.assert_equal(gt / (region[0] * region[1]), res)

        # Not a divisor
        self.assertRaises(RuntimeError, ip.bin_image, cl_im, (4, 10))
        self.assertRaises(RuntimeError, ip.bin_image, cl_im, (5, 8))
        self.assertRaises(RuntimeError, ip.bin_image, cl_im, (4, 7), offset=(2, 2))

    def test_decimate(self):
        n = 16
        sigma = fwnm_to_sigma(1)
        shape = (n // 2, n // 2)

        image = np.arange(n * n).reshape(n, n).astype(cfg.PRECISION.np_float) // n ** 2
        fltr = get_gauss_2d((n, n), sigma, fourier=True)
        filtered = np.fft.ifft2(np.fft.fft2(image) * fltr).real
        gt = bin_cpu(filtered, shape)

        res = ip.decimate(image, shape, sigma=sigma, average=False).get()
        np.testing.assert_almost_equal(gt, res, decimal=6)

        # With averaging
        res = ip.decimate(image, shape, sigma=sigma, average=True).get()
        gt = gt / 4
        np.testing.assert_almost_equal(gt, res, decimal=6)

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_rescale(self):
        orig_shape = 8, 4
        shape = 4, 8
        image = (
            np.arange(orig_shape[0] * orig_shape[1])
            .reshape(orig_shape)
            .astype(cfg.PRECISION.np_float)
        )
        res = ip.rescale(image, shape).get()
        gt = rescale_scipy(image, (0.5, 2))

        np.testing.assert_almost_equal(res, gt)

        cfg.PRECISION.set_precision(True)
        image = image.astype(cfg.PRECISION.np_float)
        self.assertRaises(TypeError, ip.rescale, image, shape)

    def test_crop(self):
        shape = 8, 4
        image = np.arange(shape[0] * shape[1]).reshape(shape).astype(cfg.PRECISION.np_float)
        x_0 = 1
        y_0 = 2
        width = 3
        height = 4
        res = ip.crop(image, (y_0, x_0, height, width)).get()

        np.testing.assert_equal(image[y_0 : y_0 + height, x_0 : x_0 + width], res)

        # Identity
        np.testing.assert_equal(image, ip.crop(image, (0, 0, shape[0], shape[1])).get())

    def test_pad(self):
        shape = 3, 2
        image = np.arange(shape[0] * shape[1]).reshape(shape).astype(cfg.PRECISION.np_float)
        x_0 = 1
        y_0 = 2
        width = 8
        height = 5
        res = ip.pad(image, (y_0, x_0, height, width)).get()

        gt = np.zeros((height, width), dtype=image.dtype)
        gt[y_0 : y_0 + shape[0], x_0 : x_0 + shape[1]] = image
        np.testing.assert_equal(gt, res)

        # Identity
        np.testing.assert_equal(image, ip.pad(image, (0, 0, shape[0], shape[1])).get())

    def test_fft(self):
        data = gpu_util.get_array(
            np.random.normal(100, 100, size=(4, 4)).astype(cfg.PRECISION.np_float)
        )
        orig = gpu_util.get_host(data)
        data = ip.fft_2(data)
        ip.ifft_2(data)
        np.testing.assert_almost_equal(orig, data.get().real, decimal=4)

        # Test double precision
        default_syris_init(double_precision=True)
        data = gpu_util.get_array(
            np.random.normal(100, 100, size=(4, 4)).astype(cfg.PRECISION.np_float)
        )
        gt = np.fft.fft2(data.get())
        data = ip.fft_2(data)
        np.testing.assert_almost_equal(gt, data.get(), decimal=4)

        gt = np.fft.ifft2(data.get())
        data = ip.ifft_2(data)
        np.testing.assert_almost_equal(gt, data.get(), decimal=4)

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_varconvolve_disk(self):
        n = 4
        shape = (n, n)
        image = np.zeros(shape, dtype=cfg.PRECISION.np_float)
        image[n // 2, n // 2] = 1
        radii = np.ones_like(image) * 1e-3
        result = ip.varconvolve_disk(image, (radii, radii), normalized=False, smooth=False).get()
        # At least one pixel in the midle must exist (just copies the original image)
        self.assertEqual(1, np.sum(result))

        radii = np.ones_like(image) * 2
        norm_result = ip.varconvolve_disk(
            image, (radii, radii), normalized=True, smooth=False
        ).get()
        self.assertAlmostEqual(1, np.sum(norm_result))

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_varconvolve_gauss(self):
        from scipy.ndimage import gaussian_filter

        n = 128
        shape = (n, n)
        image = np.zeros(shape, dtype=cfg.PRECISION.np_float)
        image[n // 2, n // 2] = 1
        sigmas = np.ones_like(image) * 1e-3
        result = ip.varconvolve_gauss(image, (sigmas, sigmas), normalized=False).get()
        # At least one pixel in the midle must exist (just copies the original image)
        self.assertEqual(1, np.sum(result))

        # Test against the scipy implementation
        sigma = fwnm_to_sigma(5, n=2)
        sigmas = np.ones_like(image) * sigma
        gt = gaussian_filter(image, sigma)
        result = ip.varconvolve_gauss(image, (sigmas, sigmas), normalized=True).get()
        np.testing.assert_almost_equal(gt, result)

    def test_compute_intensity(self):
        shape = (32, 32)
        u = (np.ones(shape) + 1j * np.ones(shape) * 3).astype(cfg.PRECISION.np_cplx)
        np.testing.assert_almost_equal(np.abs(u) ** 2, ip.compute_intensity(u).get())

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_rescale_up(self):
        # Use spline and not zoom or imresize because they don't behave exactly as we define
        n = 8
        square = np.zeros((n, n), dtype=np.float32)
        square[2:-2, 2:-2] = 1

        # Same
        res = ip.rescale(square, (n, n)).get()
        np.testing.assert_almost_equal(res, square)

        # Various odd/even combinations in the x, y directions
        for (ss_y, ss_x) in itertools.product((2, 3), (2, 3)):
            hd_n = ss_x * n
            hd_m = ss_y * n
            res = ip.rescale(square, (hd_m, hd_n)).get()
            gt = rescale_scipy(square, (ss_y, ss_x)).astype(cfg.PRECISION.np_float)
            np.testing.assert_almost_equal(res, gt, decimal=2)

    @unittest.skipIf(not are_images_supported(), "Images not supported")
    def test_rescale_down(self):
        # Use spline and not zoom or imresize because they don't behave exactly as we define
        n = 18
        square = np.zeros((n, n), dtype=np.float32)
        square[4:-4, 4:-4] = 1

        # Same
        res = ip.rescale(square, (n, n)).get()
        np.testing.assert_almost_equal(res, square)

        # Various odd/even combinations in the x, y directions
        for (ss_y, ss_x) in itertools.product((1.0 / 2, 1.0 / 3), (1.0 / 2, 1.0 / 3)):
            hd_n = int(ss_x * n)
            hd_m = int(ss_y * n)
            res = ip.rescale(square, (hd_m, hd_n)).get()
            gt = rescale_scipy(square, (ss_y, ss_x))
            np.testing.assert_almost_equal(res, gt, decimal=2)
