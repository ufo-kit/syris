import numpy as np
import pyopencl as cl
from syris.gpu import util as gpu_util
from syris import config as cfg
from unittest import TestCase


class TestVComplex(TestCase):

    def _execute_kernel(self, kernel):
        kernel(self.queue, (1,), None, self.mem_0, self.mem_1, self.mem_out)

    def setUp(self):
        self.ctx = gpu_util.get_cuda_context()
        self.queue = gpu_util.get_command_queues(self.ctx)[0]
        self.num_0 = np.array([17 - 38j], dtype=cfg.NP_CPLX)
        self.num_1 = np.array([-135 + 563j], dtype=cfg.NP_CPLX)
        self.mem_0 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE |
                               cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.num_0)
        self.mem_1 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE |
                               cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.num_1)
        self.mem_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 cfg.CL_CPLX)
        self.host_array = np.empty(1, cfg.NP_CPLX)
        src = gpu_util.get_source(["complex.cl"])
        self.prg = cl.Program(self.ctx, src).build()

    def test_addition(self):
        self._execute_kernel(self.prg.vc_add_kernel)
        cl.enqueue_copy(self.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 + self.num_1)

    def test_subtraction(self):
        self._execute_kernel(self.prg.vc_sub_kernel)
        cl.enqueue_copy(self.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 - self.num_1)

    def test_multiplication(self):
        self._execute_kernel(self.prg.vc_mul_kernel)
        cl.enqueue_copy(self.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 * self.num_1)

    def test_division(self):
        self._execute_kernel(self.prg.vc_div_kernel)
        cl.enqueue_copy(self.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 / self.num_1)
