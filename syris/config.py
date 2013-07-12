"""
Utility functions.
"""
import numpy as np
import pyopencl as cl
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

# OpenCL functions which are wrapped for profiling if profiling is enabled.
PROFILED_CL_FUNCTIONS = [cl.enqueue_nd_range_kernel, cl.enqueue_copy]


def single_precision():
    """Return True if single precision is set for floating point numbers."""
    global CL_FLOAT

    return CL_FLOAT == 4
