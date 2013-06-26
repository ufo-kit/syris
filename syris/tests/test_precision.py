import numpy as np
import pyopencl as cl
from syris import config as cfg
from syris.gpu import util as gpu_util
import unittest

ctx = gpu_util.get_cuda_context()
queue = gpu_util.get_command_queues(ctx)[0]


class TestPrecision(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.kernel_fn = "float_test.cl"

    def _create_mem_objs(self, ctx, n):
        mem = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n*cfg.cl_float)
        ar = np.empty(n, dtype=cfg.np_float)

        return mem, ar

    def _execute_and_check(self):
        prg = cl.Program(ctx, gpu_util.get_source(self.kernel_fn)).build()
        mem, ar = self._create_mem_objs(ctx, self.n)
        prg.float_test(queue,
                       (self.n,),
                       None,
                       mem)
        cl.enqueue_copy(queue, ar, mem)
        res = ar == np.array([0, 1], dtype=cfg.np_float)
        self.assertTrue(res.all())

    def test_float(self):
        self._execute_and_check()

    def test_double(self):
        cfg.cl_float = 8
        cfg.np_float = np.float64
        self._execute_and_check()
