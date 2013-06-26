"""
Utility functions.
"""
import atexit
from optparse import OptionParser
import logging
import numpy as np
from syris import profiling as prf
from syris.profiling import Profiler, DummyProfiler


logger = logging.getLogger()

# Default single precision specification of data types.
np_float = np.float32
np_cplx = np.complex64
# In bytes.
cl_float = 4
cl_cplx = 8

# Available for customization.
parser = OptionParser()

_initialized = False


def init(queues):
    global _initialized
    if _initialized:
        raise RuntimeError("Already initialized.")
    _initialized = True

    parser.add_option("-d", "--double", dest="double", action="store_true",
                      default=False,
                      help="Use double precision", metavar="DOUBLE")
    parser.add_option("-l", "--log", dest="logging_level",
                      help="logging level", metavar="LEVEL")
    parser.add_option("--profile", dest="profiler", action="count",
                      default=0, help="enable profiling", metavar="PROFILE")
    parser.add_option("--profiler-file", dest="profiler_file", action="store",
                      default="profile.dat", help="profiler file",
                      metavar="PROFILER_FILE")
    parser.add_option("-o", "--logger-file", dest="logger_file",
                      default="simulation.log",
                      help="log file path", metavar="LOG_FILE")

    cmdoptions = parser.parse_args()[0]

    if cmdoptions.double:
        # redefine floating point data types to double precision
        global np_float
        global np_cplx
        global cl_float
        global cl_cplx

        np_float = np.float64
        np_cplx = np.complex128
        cl_float = 8
        cl_cplx = 16

    if cmdoptions.logging_level is not None:
        init_logging(logging.getLevelName(cmdoptions.logging_level),
                     cmdoptions.logger_file)

    if cmdoptions.profiler:
        prf.profiler = Profiler(queues, cmdoptions.profiler_file)
        prf.profiler.start()

        @atexit.register
        def exit_handler():
            logger.info("Shutting down profiler...")
            prf.profiler.shutdown()
    else:
        prf.profiler = DummyProfiler()


def init_logging(level, logger_file):
    """Initialize logging with output to *logger_file*."""
    logger.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    fh = logging.FileHandler(logger_file, "w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
