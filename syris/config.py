"""
Utility functions.
"""
import atexit
from optparse import OptionParser
import logging
import numpy as np
from syris import profiling as prf
from syris.profiling import Profiler, DummyProfiler


LOGGER = logging.getLogger()

# Default single precision specification of data types.
NP_FLOAT = np.float32
NP_CPLX = np.complex64
# Bytes per value.
CL_FLOAT = 4
CL_CPLX = 8

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# Available for customization.
PARSER = OptionParser()

_INITIALIZED = False


def init(queues=None):
    """Initialization of the framework. This method *must* be called before
    the simulation calculation.
    """
    global _INITIALIZED
    if _INITIALIZED:
        raise RuntimeError("Already initialized.")
    _INITIALIZED = True

    PARSER.add_option("-d", "--double", dest="double", action="store_true",
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

    cmdoptions = PARSER.parse_args()[0]

    # Single or double floating point precision settings.
    if cmdoptions.double:
        # redefine floating point data types to double precision
        global NP_FLOAT
        global NP_CPLX
        global CL_FLOAT
        global CL_CPLX

        NP_FLOAT = np.float64
        NP_CPLX = np.complex128
        CL_FLOAT = 8
        CL_CPLX = 16

    # Logging level and output file.
    if cmdoptions.logging_level is not None:
        _init_logging(logging.getLevelName(cmdoptions.logging_level),
                      cmdoptions.logger_file)

    # Profiling options, they depend on the created command queues.
    if cmdoptions.profiler:
        prf.PROFILER = Profiler(queues, cmdoptions.profiler_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            LOGGER.debug("Shutting down profiler...")
            prf.PROFILER.shutdown()
    else:
        prf.PROFILER = DummyProfiler()

    # Set the pmasf excutable path.
    global PMASF_FILE
    if cmdoptions.pmasf_file != PMASF_FILE:
        PMASF_FILE = cmdoptions.pmasf_file


def _init_logging(level, logger_file):
    """Initialize logging with output to *logger_file*."""
    LOGGER.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    file_handler = logging.FileHandler(logger_file, "w")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    LOGGER.addHandler(file_handler)
    LOGGER.debug("Log.")
