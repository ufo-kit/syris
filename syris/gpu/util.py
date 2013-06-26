"""
Utility functions concerning GPU programming.
"""

import pyopencl as cl
import time
from syris import profiling as prf
from syris import config as cfg
import logging
import os


LOGGER = logging.getLogger(__name__)


SINGLE_HEADER = """
        typedef float vfloat;
        typedef float2 vfloat2;
        typedef float3 vfloat3;
        typedef float4 vfloat4;
        typedef float8 vfloat8;
        typedef float16 vfloat16;

        """

DOUBLE_HEADER = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        typedef double vfloat;
        typedef double2 vfloat2;
        typedef double3 vfloat3;
        typedef double4 vfloat4;
        typedef double8 vfloat8;
        typedef double16 vfloat16;

        """


def get_source(file_name, precision_sensitive=True):
    """Get source from a file with *file_name* and apply single or double
    precision parametrization if *precision_sensitive* is True.
    """
    path = os.path.join(os.path.dirname(__file__), "opencl", file_name)
    string = open(path, "r").read()

    if precision_sensitive:
        if cfg.CL_FLOAT == 4:
            string = SINGLE_HEADER + string
        else:
            string = DOUBLE_HEADER + string

    return string


def execute(function, *ARGS, **kwargs):
    """Execute a *function* which can be an OpenCL kernel or other OpenCL
    related function and profile it.
    """
    event = function(*ARGS, **kwargs)
    if function.__class__ == cl.Kernel:
        func_name = function.function_name
    else:
        func_name = function.__name__

    prf.PROFILER.add(event, func_name)

    return event


def get_cuda_platform(platforms):
    """Get the NVIDIA CUDA platform if any."""
    for plat in platforms:
        if plat.name == "NVIDIA CUDA":
            return plat
    return None


def get_cuda_devices():
    """Get all CUDA devices."""
    return get_cuda_platform(cl.get_platforms()).get_devices()


def get_cuda_context(devices=None, properties=None):
    """Create an NVIDIA CUDA context with *properties* for *devices*,
    if None are given create the context for all available."""
    if devices is None:
        devices = get_cuda_platform(cl.get_platforms()).get_devices()

    LOGGER.debug("Creating OpenCL context for %d devices." % (len(devices)))
    start = time.time()
    ctx = cl.Context(devices, properties)
    LOGGER.debug("OpenCL context created in %g s." % (time.time() - start))

    return ctx


def get_command_queues(context, devices=None,
                       queue_args=(), queue_kwargs={}):
    """Create command queues for each of the *devices* within a specified
    *context*. If *devices* is None, NVIDIA GPUs are automatically
    detected and used for creating the command queues.
    """
    if devices is None:
        devices = get_cuda_devices()

    LOGGER.debug("Creating %d command queues." % (len(devices)))
    queues = []
    for device in devices:
        queues.append(cl.CommandQueue(context, device,
                                      *queue_args, **queue_kwargs))

    LOGGER.debug("%d command queues created." % (len(devices)))

    return queues
