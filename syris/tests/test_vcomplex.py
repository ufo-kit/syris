import numpy as np
import pyopencl as cl
import syris
from syris.gpu import util as gpu_util
from syris import config as cfg
from syris.tests import SyrisTest, opencl, slow


@opencl
@slow
class TestVComplex(SyrisTest):

    def _execute_kernel(self, kernel):
        kernel(cfg.OPENCL.queue, (1,), None, self.mem_0, self.mem_1, self.mem_out)

    def setUp(self):
        syris.init(device_index=0)
        self.num_0 = np.array([17 - 38j], dtype=cfg.PRECISION.np_cplx)
        self.num_1 = np.array([-135 + 563j], dtype=cfg.PRECISION.np_cplx)
        self.mem_0 = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                               cl.mem_flags.COPY_HOST_PTR, hostbuf=self.num_0)
        self.mem_1 = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                               cl.mem_flags.COPY_HOST_PTR, hostbuf=self.num_1)
        self.mem_out = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE, cfg.PRECISION.cl_cplx)
        self.host_array = np.empty(1, cfg.PRECISION.np_cplx)
        src = gpu_util.get_source(["vcomplex.cl"])
        self.prg = cl.Program(cfg.OPENCL.ctx, src).build()

    def test_addition(self):
        self._execute_kernel(self.prg.vc_add_kernel)
        cl.enqueue_copy(cfg.OPENCL.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 + self.num_1)

    def test_subtraction(self):
        self._execute_kernel(self.prg.vc_sub_kernel)
        cl.enqueue_copy(cfg.OPENCL.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 - self.num_1)

    def test_multiplication(self):
        self._execute_kernel(self.prg.vc_mul_kernel)
        cl.enqueue_copy(cfg.OPENCL.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 * self.num_1)

    def test_division(self):
        self._execute_kernel(self.prg.vc_div_kernel)
        cl.enqueue_copy(cfg.OPENCL.queue, self.host_array, self.mem_out)
        self.assertAlmostEqual(self.host_array[0], self.num_0 / self.num_1)
