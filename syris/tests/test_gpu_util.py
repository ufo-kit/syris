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

import itertools
import numpy as np
import pyopencl as cl
from syris import config as cfg
from syris.gpu import util as gu
from syris.tests import default_syris_init, SyrisTest


def _has_platform_type(device_type):
    for platform in cl.get_platforms():
        device = platform.get_devices()[0]
        if device.type == device_type:
            return True
    return False


class TestGPUUtil(SyrisTest):
    def setUp(self):
        default_syris_init(profiling=True)
        self.data = np.arange(10).astype(cfg.PRECISION.np_float)
        self.mem = cl.Buffer(
            cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data
        )

    def tearDown(self):
        del self.mem

    def test_cache(self):
        self.assertEqual(
            gu.cache(self.mem, self.data.shape, cfg.PRECISION.np_float, cfg.CACHE_DEVICE), self.mem
        )
        host_cache = gu.cache(self.mem, self.data.shape, self.data.dtype, cfg.CACHE_HOST)

        np.testing.assert_equal(self.data, host_cache)

    def test_get_cache(self):
        self.assertEqual(gu.get_cache(self.mem), self.mem)

        mem = gu.get_cache(self.data)
        res = np.empty(self.data.shape, dtype=self.data.dtype)
        cl.enqueue_copy(cfg.OPENCL.queue, res, mem)
        np.testing.assert_equal(self.data, res)

    def test_conversion(self):
        def _test():
            shape = 8, 4
            dtypes = ["i", "u", "f"]
            lengths = [2, 4, 8]
            types = [
                np.dtype("{}{}".format(dt, length))
                for dt, length in itertools.product(dtypes, lengths)
            ]
            types.append(np.dtype("i1"))
            types.append(np.dtype("u1"))
            types += [np.dtype("c8"), np.dtype("c16")]
            for dtype in types:
                np_data = np.arange(shape[0] * shape[1]).reshape(shape).astype(dtype)
                # host -> Array
                cl_data = gu.get_array(np_data)
                np.testing.assert_equal(np_data, cl_data.get())
                # Array -> Array
                res = gu.get_array(cl_data)
                np.testing.assert_equal(res.get(), cl_data.get())
                # Array -> host
                host_data = gu.get_host(cl_data)
                np.testing.assert_equal(np_data, host_data)
                # host -> host
                host_data = gu.get_host(np_data)
                np.testing.assert_equal(np_data, host_data)
                if gu.are_images_supported() and dtype.kind != "c":
                    # numpy -> Image and Image -> Array
                    image = gu.get_image(np_data)
                    back = gu.get_array(image).get()
                    np.testing.assert_equal(back, np_data)
                    # Image -> host
                    host_data = gu.get_host(image)
                    np.testing.assert_equal(host_data, np_data)
                    # Array -> Image
                    image = gu.get_image(cl_data)
                    back = gu.get_array(image).get()
                    np.testing.assert_equal(back, np_data)
                    # Image -> Image
                    image_2 = gu.get_image(image)
                    back = gu.get_array(image_2).get()
                    np.testing.assert_equal(back, np_data)

        # Single precision
        _test()

        # Double precision
        cfg.PRECISION.set_precision(True)
        _test()

        # Unrecognized data types
        self.assertRaises(TypeError, gu.get_array, 1)
        self.assertRaises(TypeError, gu.get_array, None)
        if gu.are_images_supported():
            self.assertRaises(TypeError, gu.get_image, 1)
            self.assertRaises(TypeError, gu.get_image, None)

            # Complex Image
            data = np.ones((4, 4), dtype=complex)
            self.assertRaises(TypeError, gu.get_image, data)

    def test_get_duration(self):
        data = np.arange(64, dtype=cfg.PRECISION.np_float)
        mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY, size=data.nbytes)

        ev = cl.enqueue_copy(cfg.OPENCL.queue, mem, data)
        ev.wait()
        gu.get_event_duration(ev)

    def test_get_platform(self):
        names = [platform.name for platform in cl.get_platforms()]
        # All present must pass
        for name in names:
            gu.get_platform(name)
        self.assertRaises(LookupError, gu.get_platform, "foo")

    def test_get_platform_by_device_type(self):
        platforms = cl.get_platforms()
        if platforms:
            device_type = platforms[0].get_devices()[0].type
            self.assertIsNotNone(gu.get_platform_by_device_type(device_type))

    def test_get_cpu_platform(self):
        if _has_platform_type(cl.device_type.CPU):
            self.assertIsNotNone(gu.get_cpu_platform())

    def test_get_gpu_platform(self):
        if _has_platform_type(cl.device_type.GPU):
            self.assertIsNotNone(gu.get_gpu_platform())
