import numpy as np
import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as g_util
from syris.tests import SyrisTest


class TestGPUSorting(SyrisTest):

    def setUp(self):
        syris.init()
        self.prg = g_util.get_program(
            g_util.get_source(["polyobject.cl",
                               "heapsort.cl"],
                              precision_sensitive=True))
        self.num = 10
        self.data = np.array([1, 8, np.nan, -1, np.nan, 8, 680, 74, 2, 0]).\
            astype(cfg.PRECISION.np_float)
        self.mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                             cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)

    def _sorted(self):
        self.prg.sort_kernel(cfg.OPENCL.queue,
                             (1,),
                             None,
                             self.mem)

        res = np.empty(self.num, cfg.PRECISION.np_float)
        cl.enqueue_copy(cfg.OPENCL.queue, res, self.mem)

        return res

    def test_normal(self):
        result = self._sorted()
        self.data.sort()
        np.testing.assert_almost_equal(self.data, result)
