import logging
import pyopencl as cl
import pyopencl.array as cl_array
import syris
import syris.physics
import syris.profiling
from syris import config as cfg
from syris.tests import opencl


@opencl
def test_init():
    syris.init(profiling=True, loglevel=logging.DEBUG, double_precision=True)
    assert logging.DEBUG == syris.physics.LOG.getEffectiveLevel()
    assert cfg.OPENCL.ctx is not None
    assert cfg.PRECISION.cl_float == 8
    assert syris.profiling.PROFILER is not None
    assert cfg.PRECISION.vfloat2 == cl_array.vec.double2


def test_no_opencl_init():
    """Initialization by broken OpenCL must work too, just the context and profiling not."""
    cl.get_platforms = lambda: None
    syris.init(profiling=False, loglevel=logging.DEBUG, double_precision=True)
    assert logging.DEBUG == syris.physics.LOG.getEffectiveLevel()
    assert cfg.PRECISION.cl_float == 8
    assert cfg.PRECISION.vfloat2 == cl_array.vec.double2
