# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""
OpenCL, logging and precision configuration. This module also defines global variables which
hold the configuration objects for precision and OpenCL. Furthermore, pmasf path is specified
here and caching policy for data as well.
"""
import logging
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as cltypes
import cupy as cp


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
            self.cp_float = cp.float64
        else:
            self.cl_float = 4
            self.cl_cplx = 8
            self.np_float = np.float32
            self.np_cplx = np.complex64
            self.cp_float = cp.float32
        self.numpy_to_opencl = {self.np_float: self.cl_float, self.np_cplx: self.cl_cplx}
        self.opencl_to_numpy = dict(
            list(zip(list(self.numpy_to_opencl.values()), list(self.numpy_to_opencl.keys())))
        )

        self.float4 = np.dtype(
            {
                'names': ['x', 'y', 'z', 'w'],
                'formats': [self.np_float] * 4,
            }
        )

        self.float2 = np.dtype(
            {
                'names': ['x', 'y'],
                'formats': [self.np_float] * 2,
            }
        )

        self.uint2 = np.dtype(
            {
                'names': ['x', 'y'],
                'formats': [np.uint32] * 2,
            }
        )

        dtype_base = "double" if double else "float"
        for i in [2, 3, 4, 8, 16]:
            setattr(self, "vfloat" + str(i), getattr(cltypes, dtype_base + str(i)))


class OpenCL(object):

    """OpenCL runtime information holder."""

    def __init__(self):
        self.ctx = None
        self.queues = []
        self.devices = []
        # Default command queue
        self.queue = None
        self.programs = {"improc": None, "physics": None, "geometry": None, "mesh": None}
        # {command queue: {shape: plan}} dictionary
        self.fft_plans = {}

class CudaPipeline:
    """
    Manage and run CUDA kernels using CuPy.
    """
    def __init__(self, headers : list, options : list = None):
        self.modules = {}
        self.kernels = {}
        self.opts = ["-I " + h + " " for h in headers]
        if options is not None:
            self.opts += options
    
    def readModuleFromFiles(self, 
        moduleName : str,
        fileNames : list, 
        options : list = None,
        name_expressions : list = None,
        backend : str = "nvcc",
        jitify : bool = False):
        if moduleName in self.modules:
            raise Exception("Module already loaded")
    
        if options is None:
            selected_options = self.opts
        else:
            selected_options = options + self.opts
        
        selected_options += ['-D__CUDA_NO_HALF_CONVERSIONS__']
        
        selected_options = tuple(selected_options,)
        print (selected_options)

        # Prepend
        code = r"""
        #include <cub/cub.cuh>
        #include <thrust/sort.h>
        #include <thrust/device_vector.h>
        #include <thrust/execution_policy.h>
        """
        for fileName in fileNames:
            with open(fileName, "r") as f:
                source = f.read()
                code += source + "\n"

        self.modules[moduleName] = cp.RawModule(
            code=code,
            options=selected_options,
            jitify=jitify,
            name_expressions=name_expressions,
            backend=backend)

    def getKernelFromModule(self, moduleName : str, kernelName : str) -> cp.RawKernel:
        if moduleName not in self.modules:
            raise Exception("Module not found")
        
        if kernelName not in self.kernels:
            self.kernels[kernelName] = self.modules[moduleName].get_function(kernelName)
        
        return self.kernels[kernelName]


def init_logging(level=logging.DEBUG, logger_file=None):
    """Initialize logging with output to *logger_file*."""
    LOG.setLevel(level)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt)

    if logger_file:
        file_handler = logging.FileHandler(logger_file, "a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        LOG.addHandler(file_handler)


PRECISION = None
OPENCL = None
CUDA_PIPELINE = None
CUDA_KERNELS = None

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# OpenCL functions which are wrapped for profiling if profiling is enabled.
PROFILED_CL_FUNCTIONS = [cl.enqueue_nd_range_kernel, cl.enqueue_copy]

# Caching constants.
CACHE_HOST = 1
CACHE_DEVICE = 2
DEFAULT_CACHE = CACHE_HOST
