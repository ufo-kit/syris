"""
Utility functions.
"""
import numpy as np
from argparse import ArgumentParser


# Default single precision specification of data types.
NP_FLOAT = np.float32
NP_CPLX = np.complex64
# Bytes per value.
CL_FLOAT = 4
CL_CPLX = 8

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# OpenCL kernels folder
KERNELS_FOLDER = "opencl"

# OpenCL executives.
CTX = None
QUEUES = None
# Default command queue.
QUEUE = None


def single_precision():
    """Return True if single precision is set for floating point numbers."""
    global CL_FLOAT

    return CL_FLOAT == 4


def get_arguments():
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
