import logging
import pyopencl.array as cl_array
import syris
import syris.physics
import syris.profiling
from syris import config as cfg


def test_init():
    syris.init(profiling=True, loglevel=logging.DEBUG, double_precision=True)
    assert logging.DEBUG == syris.physics.LOG.getEffectiveLevel()
    assert cfg.OPENCL.ctx is not None
    assert cfg.PRECISION.cl_float == 8
    assert syris.profiling.PROFILER is not None
    assert cfg.PRECISION.vfloat2 == cl_array.vec.double2
