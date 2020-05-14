"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""
import atexit
import logging
import pyopencl as cl
import syris.config as cfg
from syris.gpu.util import make_opencl_defaults, init_programs, execute_profiled
from syris import profiling as prf


__version__ = "0.3dev"


LOG = logging.getLogger(__name__)


def init(
    platform_name=None,
    device_type=None,
    device_index=None,
    profiling=False,
    profiling_file="profile.dat",
    loglevel=logging.INFO,
    logfile=None,
    double_precision=False,
):
    """Initialize syris with *device_index*."""
    cfg.init_logging(level=loglevel, logger_file=logfile)
    cfg.PRECISION = cfg.Precision(double_precision)
    cfg.OPENCL = cfg.OpenCL()
    platforms = []
    try:
        platforms = cl.get_platforms()
    except Exception as e:
        LOG.exception(str(e))
    else:
        if not platforms:
            LOG.warning("No OpenCL platforms found, GPU computing will not be available")
        else:
            make_opencl_defaults(
                platform_name=platform_name,
                device_type=device_type,
                device_index=device_index,
                profiling=profiling,
            )
    if profiling:
        _wrap_opencl()
        prf.PROFILER = prf.Profiler(cfg.OPENCL.queues, profiling_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            prf.PROFILER.shutdown()

    if platforms:
        init_programs()


def _wrap_opencl():
    for function in cfg.PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, execute_profiled(function))
