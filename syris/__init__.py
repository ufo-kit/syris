"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""
import atexit
import logging
import pyopencl as cl
from syris.config import OPENCL, PRECISION, PROFILED_CL_FUNCTIONS, init_logging
from syris.gpu.util import make_opencl_defaults, init_programs, execute_profiled
from syris import profiling as prf


__version__ = '0.1'


def init(device_index=0, profiling=True, profiling_file='profile.dat', loglevel=logging.DEBUG,
         logfile=None, double_precision=False):
    """Initialize syris with *device_index*."""
    PRECISION.set_precision(double_precision)
    if OPENCL.ctx is None:
        make_opencl_defaults(device_index=device_index, profiling=profiling)
    if profiling:
        _wrap_opencl()
        prf.PROFILER = prf.Profiler(OPENCL.queues, profiling_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            prf.PROFILER.shutdown()

    init_logging(level=loglevel, logger_file=logfile)
    init_programs()


def _wrap_opencl():
    for function in PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, execute_profiled(function))
