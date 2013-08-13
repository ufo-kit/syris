"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""

__version__ = 0.1


import atexit
import syris.config as cfg
import logging
import numpy as np
import pyopencl as cl
from syris import profiling as prf
from syris.profiling import Profiler, DummyProfiler
from syris.gpu import util as g_util
from syris import physics, imageprocessing
from argparse import ArgumentParser

LOGGER = logging.getLogger()


def _get_arguments():
    parser = ArgumentParser()

    parser.add_argument("--double", dest="double", action="store_true",
                        default=False, help="Use double precision")
    parser.add_argument("-l", "--log", dest="logging_level",
                        help="logging level")
    parser.add_argument("--profile", dest="profiler", action="count",
                        default=0, help="enable profiling")
    parser.add_argument("--profiler-file", dest="profiler_file",
                        action="store",
                        default="profile.dat", help="profiler file")
    parser.add_argument("--LOGGER-file", dest="logger_file",
                        default="simulation.log", help="log file path")
    parser.add_argument("--pmasf-file", dest="pmasf_file", default="pmasf",
                        help="full path to the pmasf binary")

    return parser.parse_known_args()[0]


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
    if profiler:
        _wrap_opencl()

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

    _init_programs()


def _wrap_opencl():
    for function in cfg.PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, g_util.execute_profiled(function))


def _init_programs():
    physics.CL_PRG = g_util.get_program(g_util.get_source(["vcomplex.cl",
                                                           "physics.cl"]))
    imageprocessing.CL_PRG = g_util.get_program(
        g_util.get_source(["vcomplex.cl", "imageprocessing.cl"]))


def _init_fp(double_prec):
    if double_prec:
        cfg.NP_FLOAT = np.float64
        cfg.NP_CPLX = np.complex128
        cfg.CL_FLOAT = 8
        cfg.CL_CPLX = 16
    else:
        cfg.NP_FLOAT = np.float32
        cfg.NP_CPLX = np.complex64
        cfg.CL_FLOAT = 4
        cfg.CL_CPLX = 8


def init(queues=None):
    """Initialize Syris by the command line arguments."""
    args = _get_arguments()
    _init_gpus(args.profiler, queues)

    # Single or double floating point precision settings.
    _init_fp(args.double)

    # Logging level and output file.
    if args.logging_level is not None:
        _init_logging(args.logging_level.upper(), args.logger_file)

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
