"""
Utility functions.
"""
import logging
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


LOG = logging.getLogger()
MAX_META_BODIES = 30


class Precision(object):

    """A precision object holds the precision settings of the floating point and complex numpy and
    OpenCL data types. If *double* is True, double precision is used.
    """

    def __init__(self, double=False):
        self.set_precision(double)

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

        dtype_base = 'double' if double else 'float'
        for i in [2, 3, 4, 8, 16]:
            setattr(self, 'vfloat' + str(i), getattr(cl_array.vec, dtype_base + str(i)))


class OpenCL(object):

    """OpenCL runtime information holder."""

    def __init__(self):
        self.ctx = None
        self.queues = []
        self.devices = []
        # Default command queue
        self.queue = None
        self.programs = {'improc': None,
                         'physics': None,
                         'geometry': None,
                         'mesh': None}
        # {command queue: {shape: plan}} dictionary
        self.fft_plans = {}


def init_logging(level=logging.DEBUG, logger_file=None):
    """Initialize logging with output to *logger_file*."""
    LOG.setLevel(level)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)

    if logger_file:
        file_handler = logging.FileHandler(logger_file, "a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        LOG.addHandler(file_handler)


PRECISION = None
OPENCL = None

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# OpenCL functions which are wrapped for profiling if profiling is enabled.
PROFILED_CL_FUNCTIONS = [cl.enqueue_nd_range_kernel, cl.enqueue_copy]

# Caching constants.
CACHE_HOST = 1
CACHE_DEVICE = 2
DEFAULT_CACHE = CACHE_HOST
