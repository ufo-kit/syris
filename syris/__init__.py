"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""
import atexit
import logging
import pyopencl as cl
from syris.config import OPENCL, PROFILED_CL_FUNCTIONS, init_logging
from syris.gpu.util import make_opencl_defaults, init_program, execute_profiled
from syris import profiling as prf


__version__ = '0.1'


def init(profiling=True, profiling_file='profile.dat', loglevel=logging.DEBUG,
         logfile=None):
    """Initialize syris."""
    if OPENCL.ctx is None:
        make_opencl_defaults(profiling=profiling)
    if profiling:
        _wrap_opencl()
        prf.PROFILER = prf.Profiler(OPENCL.queues, profiling_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            prf.PROFILER.shutdown()

    init_logging(level=loglevel, logger_file=logfile)
    init_program()


def _wrap_opencl():
    for function in PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, execute_profiled(function))
