"""
Utility functions concerning GPU programming.
"""

import pyopencl as cl
import time
from syris import profiling as prf
from syris import config as cfg
import logging
import os


logger = logging.getLogger(__name__)


single_header = """
        typedef float vfloat;
        typedef float2 vfloat2;
        typedef float3 vfloat3;
        typedef float4 vfloat4;
        typedef float8 vfloat8;
        typedef float16 vfloat16;

        """

double_header = """
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
    s = open(path, "r").read()

    if precision_sensitive:
        if cfg.cl_float == 4:
            s = single_header + s
        else:
            s = double_header + s

    return s


def execute(function, *args, **kwargs):
    """Execute a *function* which can be an OpenCL kernel or other OpenCL
    related function and profile it.
    """
    event = function(*args, **kwargs)
    if function.__class__ == cl.Kernel:
        func_name = function.function_name
    else:
        func_name = function.__name__

    prf.profiler.add(event, func_name)

    return event


def get_cuda_platform(platforms):
    for p in platforms:
        if p.name == "NVIDIA CUDA":
            return p
    return None


def get_cuda_context():
    platforms = cl.get_platforms()
    p = get_cuda_platform(platforms)
    devices = p.get_devices()

    logger.debug("Creating OpenCL context for %d devices." % (len(devices)))
    st = time.time()
    ctx = cl.Context(devices)
    logger.debug("OpenCL context created in %g s." % (time.time()-st))

    return ctx


def get_cuda_devices():
    """Get all CUDA devices."""
    return get_cuda_platform(cl.get_platforms()).get_devices()


def get_command_queues(context, devices=None,
                       queue_args=(), queue_kwargs={}):
    """Create command queues for each of the *devices* within a specified
    *context*. If *devices* is None, NVIDIA GPUs are automatically
    detected and used for creating the command queues.
    """
    if devices is None:
        devices = get_cuda_devices()

    logger.debug("Creating %d command queues." % (len(devices)))
    queues = []
    for device in devices:
        queues.append(cl.CommandQueue(context, device,
                                      *queue_args, **queue_kwargs))

    logger.debug("%d command queues created." % (len(devices)))

    return queues
