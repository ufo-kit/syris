"""
Utility functions.
"""
import logging
import numpy as np
import pyopencl as cl


LOG = logging.getLogger()


class Precision(object):

    """A precision object holds information about the precision
    of the floating point and complex numpy and OpenCL data types.
    """

    def __init__(self):
        self.set_precision(False)

    def is_single(self):
        """Return True if the precision is single."""
        return self.cl_float == 4

    def set_precision(self, double):
        """If *double* is True set the double precision."""
        if double:
            self.cl_float = 8
            self.cl_cplx = 16
            self.np_float = np.float64
            self.np_cplx = np.complex128
        else:
            self.cl_float = 4
            self.cl_cplx = 8
            self.np_float = np.float32
            self.np_cplx = np.complex64
        self.numpy_to_opencl = {self.np_float: self.cl_float, self.np_cplx: self.cl_cplx}
        self.opencl_to_numpy = dict(zip(self.numpy_to_opencl.values(),
                                        self.numpy_to_opencl.keys()))


class OpenCL(object):

    """OpenCL runtime information holder."""

    def __init__(self):
        self.kernels_dir = 'opencl'
        self.ctx = None
        self.queues = []
        self.devices = []
        # Default command queue
        self.queue = None
        self.program = None


def init_logging(level=logging.DEBUG, logger_file=None):
    """Initialize logging with output to *logger_file*."""
    LOG.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    if logger_file:
        file_handler = logging.FileHandler(logger_file, "w")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        LOG.addHandler(file_handler)


PRECISION = Precision()
OPENCL = OpenCL()

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# OpenCL functions which are wrapped for profiling if profiling is enabled.
PROFILED_CL_FUNCTIONS = [cl.enqueue_nd_range_kernel, cl.enqueue_copy]

# Caching constants.
CACHE_HOST = 1
CACHE_DEVICE = 2
DEFAULT_CACHE = CACHE_HOST
