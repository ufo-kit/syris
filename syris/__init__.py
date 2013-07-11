"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""

__version__ = 0.1


import atexit
import config as cfg
import logging
import syris.profiling as prf
from syris.profiling import Profiler, DummyProfiler
from syris.gpu import util as g_util
from syris import physics

LOGGER = logging.getLogger()


def _init_logging(level, logger_file):
    """Initialize logging with output to *logger_file*."""
    LOGGER.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    file_handler = logging.FileHandler(logger_file, "w")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(file_handler)


def _init_gpus(profiler, queues=None):
    import pyopencl as cl

    if queues is None:
        cfg.CTX = g_util.get_cuda_context()

        if profiler:
            kwargs = {"properties":
                      cl.command_queue_properties.PROFILING_ENABLE}
        else:
            kwargs = {}

        cfg.QUEUES = g_util.get_command_queues(cfg.CTX, queue_kwargs=kwargs)
        cfg.QUEUE = cfg.QUEUES[0]
    else:
        if profiler and not queues[0].properties & \
                cl.command_queue_properties.PROFILING_ENABLE:
            raise ValueError("Command queues are not enabled for profiling.")

        cfg.QUEUES = queues
        cfg.QUEUE = cfg.QUEUES[0]

    physics.CL_PRG = g_util.get_program(g_util.get_source(["vcomplex.cl",
                                                           "physics.cl"]))


def init(queues=None):
    """Initialize Syris by the command line arguments."""
    import numpy as np

    args = cfg.get_arguments()
    _init_gpus(args.profiler, queues)

    # Single or double floating point precision settings.
    if args.double:
        # redefine floating point data types to double precision
        cfg.NP_FLOAT = np.float64
        cfg.NP_CPLX = np.complex128
        cfg.CL_FLOAT = 8
        cfg.CL_CPLX = 16

    # Logging level and output file.
    if args.logging_level is not None:
        _init_logging(logging.getLevelName(args.logging_level),
                      args.logger_file)

    # Profiling options, they depend on the created command queues.
    if args.profiler:
        prf.PROFILER = Profiler(cfg.QUEUES, args.profiler_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            LOGGER.debug("Shutting down profiler...")
            prf.PROFILER.shutdown()
    else:
        prf.PROFILER = DummyProfiler()

    # Set the pmasf excutable path.
    if args.pmasf_file != cfg.PMASF_FILE:
        cfg.PMASF_FILE = args.pmasf_file
