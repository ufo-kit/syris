import numpy as np
import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as g_util
from unittest import TestCase


class TestGPUSorting(TestCase):

    def setUp(self):
        syris.init()
        self.prg = g_util.get_program(
            g_util.get_source(["polyobject.cl",
                               "heapsort.cl"],
                              precision_sensitive=True))
        self.num = 10
        self.data = np.array([1, 8, np.nan, -1, np.nan, 8, 680, 74, 2, 0]).\
            astype(cfg.NP_FLOAT)
        self.mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE |
                             cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)

    def _sorted(self):
        self.prg.sort_kernel(cfg.QUEUE,
                             (1,),
                             None,
                             self.mem)

        res = np.empty(self.num, cfg.NP_FLOAT)
        cl.enqueue_copy(cfg.QUEUE, res, self.mem)

        return res

    def test_normal(self):
        result = self._sorted()
        self.data.sort()
        np.testing.assert_almost_equal(self.data, result)
