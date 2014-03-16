import numpy as np
import pyopencl as cl
from syris.config import PRECISION
from syris.gpu import util as gpu_util
from syris.tests.base import SyrisTest

ctx = gpu_util.get_cuda_context()
queue = gpu_util.get_command_queues(ctx)[0]


class TestPrecision(SyrisTest):

    def setUp(self):
        self.n = 2
        self.kernel_fn = "vfloat_test.cl"

    def _create_mem_objs(self, ctx, n):
        mem = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n * PRECISION.cl_float)
        ar = np.empty(n, dtype=PRECISION.np_float)

        return mem, ar

    def _execute_and_check(self):
        prg = cl.Program(ctx, gpu_util.get_source([self.kernel_fn])).build()
        mem, ar = self._create_mem_objs(ctx, self.n)
        prg.float_test(queue,
                       (self.n,),
                       None,
                       mem)
        cl.enqueue_copy(queue, ar, mem)
        res = ar == np.array([0, 1], dtype=PRECISION.np_float)
        self.assertTrue(res.all())

    def test_float(self):
        self._execute_and_check()

    def test_double(self):
        PRECISION.set_precision(True)
        self._execute_and_check()
