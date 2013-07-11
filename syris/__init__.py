"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""

__version__ = 0.1

import atexit
import logging
import numpy as np
from optparse import OptionParser
import pyopencl as cl
import config as cfg
import profiling as prf
from profiling import Profiler, DummyProfiler
from gpu import util as g_util


LOGGER = logging.getLogger()


PARSER = OptionParser()

PARSER.add_option("--double", dest="double", action="store_true",
                  default=False,
                  help="Use double precision", metavar="DOUBLE")
PARSER.add_option("-l", "--log", dest="logging_level",
                  help="logging level", metavar="LEVEL")
PARSER.add_option("--profile", dest="profiler", action="count",
                  default=0, help="enable profiling", metavar="PROFILE")
PARSER.add_option("--profiler-file", dest="profiler_file", action="store",
                  default="profile.dat", help="profiler file",
                  metavar="PROFILER_FILE")
PARSER.add_option("--LOGGER-file", dest="logger_file",
                  default="simulation.log",
                  help="log file path", metavar="LOG_FILE")
PARSER.add_option("--pmasf-file", dest="pmasf_file", default="pmasf",
                  help="full path to the pmasf binary",
                  metavar="PMASF_FILE")

CMDOPTIONS = PARSER.parse_args()[0]


def _init_logging(level, logger_file):
    """Initialize logging with output to *logger_file*."""
    LOGGER.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    file_handler = logging.FileHandler(logger_file, "w")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(file_handler)


def _init_command_queues():
    if CMDOPTIONS.profiler:
        kwargs = {"properties": cl.command_queue_properties.PROFILING_ENABLE}
    else:
        kwargs = {}

    cfg.QUEUES = g_util.get_command_queues(cfg.CTX, queue_kwargs=kwargs)
    cfg.QUEUE = cfg.QUEUES[0]


# Single or double floating point precision settings.
if CMDOPTIONS.double:
    # redefine floating point data types to double precision
    cfg.NP_FLOAT = np.float64
    cfg.NP_CPLX = np.complex128
    cfg.CL_FLOAT = 8
    cfg.CL_CPLX = 16

# Logging level and output file.
if CMDOPTIONS.logging_level is not None:
    _init_logging(logging.getLevelName(CMDOPTIONS.logging_level),
                  CMDOPTIONS.logger_file)

cfg.CTX = g_util.get_cuda_context()
_init_command_queues()

# Profiling options, they depend on the created command queues.
if CMDOPTIONS.profiler:
    prf.PROFILER = Profiler(cfg.QUEUES, CMDOPTIONS.profiler_file)
    prf.PROFILER.start()

    @atexit.register
    def exit_handler():
        """Shutdown the profiler on exit."""
        LOGGER.debug("Shutting down profiler...")
        prf.PROFILER.shutdown()
else:
    prf.PROFILER = DummyProfiler()

# Set the pmasf excutable path.
if CMDOPTIONS.pmasf_file != cfg.PMASF_FILE:
    cfg.PMASF_FILE = CMDOPTIONS.pmasf_file
