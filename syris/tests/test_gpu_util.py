import numpy as np
import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as gu
from syris.tests import SyrisTest


class TestGPUUtil(SyrisTest):

    def setUp(self):
        syris.init()
        self.data = np.arange(10).astype(cfg.PRECISION.np_float)
        self.mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                             cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)

    def test_cache(self):
        self.assertEqual(gu.cache(self.mem, self.data.shape, cfg.PRECISION.np_float,
                                  cfg.CACHE_DEVICE), self.mem)
        host_cache = gu.cache(self.mem, self.data.shape, self.data.dtype,
                              cfg.CACHE_HOST)

        np.testing.assert_equal(self.data, host_cache)

    def test_get_cache(self):
        self.assertEqual(gu.get_cache(self.mem), self.mem)

        mem = gu.get_cache(self.data)
        res = np.empty(self.data.shape, dtype=self.data.dtype)
        cl.enqueue_copy(cfg.OPENCL.queue, res, mem)
        np.testing.assert_equal(self.data, res)