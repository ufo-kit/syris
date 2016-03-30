"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""
import atexit
import logging
import pyopencl as cl
import syris.config as cfg
from syris.gpu.util import make_opencl_defaults, init_programs, execute_profiled
from syris import profiling as prf


__version__ = '0.1'


def init(platform_name=None, device_index=None, profiling=True, profiling_file='profile.dat',
         loglevel=logging.DEBUG, logfile=None, double_precision=False):
    """Initialize syris with *device_index*."""
    cfg.init_logging(level=loglevel, logger_file=logfile)
    cfg.PRECISION = cfg.Precision(double_precision)
    cfg.OPENCL = cfg.OpenCL()
    make_opencl_defaults(platform_name=platform_name, device_index=device_index, profiling=profiling)
    if profiling:
        _wrap_opencl()
        prf.PROFILER = prf.Profiler(cfg.OPENCL.queues, profiling_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            prf.PROFILER.shutdown()

    init_programs()


def _wrap_opencl():
    for function in cfg.PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, execute_profiled(function))
