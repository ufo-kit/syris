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
import pyopencl as cl
import syris.config as cfg
from syris.gpu import util as gpu_util
from syris.tests import default_syris_init, SyrisTest


class TestPrecision(SyrisTest):
    def setUp(self):
        default_syris_init()
        self.n = 2
        self.kernel_fn = "vfloat_test.cl"

    def _create_mem_objs(self, n):
        mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE, size=n * cfg.PRECISION.cl_float)
        ar = np.empty(n, dtype=cfg.PRECISION.np_float)

        return mem, ar

    def _execute_and_check(self):
        prg = cl.Program(cfg.OPENCL.ctx, gpu_util.get_source([self.kernel_fn])).build()
        mem, ar = self._create_mem_objs(self.n)
        prg.float_test(cfg.OPENCL.queue, (self.n,), None, mem)
        cl.enqueue_copy(cfg.OPENCL.queue, ar, mem)
        res = ar == np.array([0, 1], dtype=cfg.PRECISION.np_float)
        self.assertTrue(res.all())

    def test_float(self):
        self._execute_and_check()

    def test_double(self):
        cfg.PRECISION.set_precision(True)
        self._execute_and_check()
