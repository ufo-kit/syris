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
from syris import config as cfg
from syris.gpu import util as g_util
from syris.tests import default_syris_init, SyrisTest


class TestGPUSorting(SyrisTest):
    def setUp(self):
        default_syris_init()
        self.prg = g_util.get_program(
            g_util.get_source(["polyobject.cl", "heapsort.cl"], precision_sensitive=True)
        )
        self.num = 10
        self.data = np.array([1, 8, np.nan, -1, np.nan, 8, 680, 74, 2, 0]).astype(
            cfg.PRECISION.np_float
        )
        self.mem = cl.Buffer(
            cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data
        )

    def _sorted(self):
        self.prg.sort_kernel(cfg.OPENCL.queue, (1,), None, self.mem)

        res = np.empty(self.num, cfg.PRECISION.np_float)
        cl.enqueue_copy(cfg.OPENCL.queue, res, self.mem)

        return res

    def test_normal(self):
        result = self._sorted()
        self.data.sort()
        np.testing.assert_almost_equal(self.data, result)
